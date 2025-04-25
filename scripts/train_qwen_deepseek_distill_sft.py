from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.trainer_utils import EvalPrediction
import torch
import argparse
import evaluate
import numpy as np
import pandas as pd
from src.data_qwen_instruct import create_llm_datasets
from src.utils import set_seed
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument('--train_path', type=str, required=True,
                       help="Path to training dataset CSV file")
    parser.add_argument('--dev_path', type=str, default="data/dev_set.csv",
                       help="Path to validation dataset CSV file")
    parser.add_argument('--max_length', type=int, default=2500)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--warmup_steps', type=int, default=100, help='Number of warmup steps')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_steps', type=int, default=300)
    parser.add_argument('--save_steps', type=int, default=300)
    parser.add_argument('--logging_steps', type=int, default=10)
    return parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(
    parse_args().model_name,
    use_fast=True,
    trust_remote_code=True,
)

# This will prevent load logits into GPU which will cause the OOM error
def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels

def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute metrics for the evaluation predictions.
    This function decodes the prediction tokens and looks for Yes/No answers after the thinking process.
    
    Args:
        eval_pred: Evaluation predictions from the model
        
    Returns:
        Dictionary of metrics including accuracy and ratio of invalid predictions
    """
    # Get predictions and references
    token_predictions = eval_pred.predictions[0]
    labels = eval_pred.label_ids
    
    pred_answers = []
    true_answers = []
    
    # Process each sequence in the batch
    for pred_seq, label_seq in zip(token_predictions, labels):
        # Filter out padding tokens
        valid_pred_tokens = pred_seq[pred_seq != -100]
        # Decode the entire prediction sequence
        decoded_pred = tokenizer.decode(valid_pred_tokens)
        
        # Find the thinking process
        parts = decoded_pred.split("</think>")
        
        if len(parts) > 1:
            # Get the text after the last </think> tag
            answer_text = parts[-1].lower().strip()
            
            # Look for Yes/No in the answer portion
            if "yes" in answer_text:
                pred_answer = 1
            elif "no" in answer_text:
                pred_answer = 0
            else:
                # Neither Yes nor No found clearly
                pred_answer = -1
        else:
            # No thinking token found
            pred_answer = -1
        
        # Process label to find true answer
        valid_indices = label_seq != -100
        if np.any(valid_indices):
            valid_labels = label_seq[valid_indices]
            # Decode the valid label tokens
            decoded_label = tokenizer.decode(valid_labels)
            
            # Look for Yes/No in the label
            if "yes" in decoded_label.lower().split():
                label_answer = 1
            else:
                label_answer = 0
            
            pred_answers.append(pred_answer)
            true_answers.append(label_answer)
    
    # Convert to numpy arrays
    pred_answers = np.array(pred_answers)
    true_answers = np.array(true_answers)
    
    # Calculate metrics
    accuracy = accuracy_score(true_answers, pred_answers)
    
    return {
        "accuracy": float(accuracy),
        "num_invalid": float(np.sum(pred_answers == -1) / len(pred_answers))
    }

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: No GPU detected. Training will be slow!")
    else:
        print(f"Using device: {device}")
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )

    
    peft_config = LoraConfig(
        r=16,  # Rank dimension - typically between 4-32
        lora_alpha=32,  # LoRA scaling factor - typically 2x rank
        lora_dropout=0.05,  # Dropout probability for LoRA layers
        bias="none",  # Bias type for LoRA. the corresponding biases will be updated during training.
        target_modules= ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],  # Which modules to apply LoRA to
        task_type="CAUSAL_LM",  # Task type for model architecture
    )
    
    # Load and prepare data
    train_df = pd.read_csv(args.train_path)
    dev_df = pd.read_csv(args.dev_path)
    
    # use a subset of the data for faster testing
    dev_df = dev_df.sample(frac=0.001)

    # Create datasets
    datasets = create_llm_datasets(
        train_df, dev_df, 
        names=["train", "dev"],
        tokenizer=tokenizer
    )
    
    train_dataset = datasets["train"]
    dev_dataset = datasets["dev"]
    
    # Create data collator for completion-only language modeling
    response_template = '<｜Assistant｜>'
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )
    
    # Training arguments
    training_args = SFTConfig(
        output_dir=f"models/{args.model_name}",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=False,  # 
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=3,
        # metric_for_best_model="eval_accuracy",
        max_seq_length = args.max_length,
        dataset_num_proc=2,
        dataset_text_field="text",
        bf16=True,
        bf16_full_eval=True,
        logging_first_step=True,
        report_to="tensorboard",
        run_name=args.model_name,
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        # callbacks=[],
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model(f"models/{args.model_name}/final_model")
    
if __name__ == "__main__":
    main()
