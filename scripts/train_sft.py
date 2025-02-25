from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import argparse
import evaluate
import numpy as np
from src.data_sft import create_llm_datasets
from src.utils import set_seed, load_data_to_df
from src.callbacks import PrinterCallback

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument('--data_dir', type=str, default="data")
    parser.add_argument('--output_dir', type=str, default="models/qwen")
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--warmup_steps', type=int, default=300, help='Number of warmup steps')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_steps', type=int, default=300)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--logging_steps', type=int, default=10)
    return parser.parse_args()


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate metrics for all classes
    accuracy = metric.compute(predictions=predictions, references=labels)
    
    # Focus on minority class (label=1)
    # Calculate binary metrics specifically for the positive class
    f1 = f1_metric.compute(predictions=predictions, references=labels, average=None)['f1'][1]
    precision = precision_metric.compute(predictions=predictions, references=labels, average=None)['precision'][1]
    recall = recall_metric.compute(predictions=predictions, references=labels, average=None)['recall'][1]
    
    # Calculate confusion matrix elements for label 1
    true_positives = np.sum((predictions == 1) & (labels == 1))
    false_positives = np.sum((predictions == 1) & (labels == 0))
    true_negatives = np.sum((predictions == 0) & (labels == 0))
    false_negatives = np.sum((predictions == 0) & (labels == 1))
    
    # Calculate additional metrics
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    balanced_accuracy = (recall + specificity) / 2
    
    return {
        'accuracy': accuracy['accuracy'],
        'balanced_accuracy': balanced_accuracy,
        'precision_class1': precision,
        'recall_class1': recall,
        'f1_class1': f1,
        'specificity': specificity,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
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
    
    tokenizer = tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    model.to(device)

    
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
    train_df, dev_df, test_df = load_data_to_df(args.data_dir)
    
    # # use a subset of the data for faster testing
    train_df = train_df.sample(frac=0.1)
    dev_df = dev_df.sample(frac=0.1)
    # test_df = test_df.sample(frac=0.001)
    
    
    train_dataset, dev_dataset, test_dataset = create_llm_datasets(train_df, dev_df, test_df, tokenizer)
    
    # Create data collator for completion-only language modeling
    collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant\n",
        tokenizer=tokenizer,
    )
    
    # Training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=3,
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        packing = False,
        max_seq_length = args.max_length,
        dataset_num_proc=2,  # why increasing this will cause BrokenPipeError: [Errno 32] Broken pipe in /.venv/lib/python3.11/site-packages/multiprocess/pool.py
        dataset_text_field="text",
        bf16=True,
        bf16_full_eval=True,
        logging_first_step=True,
        report_to="tensorboard",
        run_name="qwen",
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
    )
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model(f"{args.output_dir}/final")
    
if __name__ == "__main__":
    main()
