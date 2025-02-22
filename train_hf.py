import os
import torch
import argparse
from transformers import (
    AutoTokenizer, 
    DataCollatorWithPadding,
    ModernBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from src.data import load_data, create_datasets
from src.utils import set_seed
import evaluate
import numpy as np
from src.callbacks import PrinterCallback

# Set tokenizers parallelism before importing transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch._dynamo
torch._dynamo.config.suppress_errors = True

def parse_args():
    parser = argparse.ArgumentParser(description='Train CDEC Resolution Model with HuggingFace Trainer')
    parser.add_argument('--model_type', type=str, choices=['encoder', 'decoder'], help='Type of model to train, either encoder or decoder')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the data files')
    parser.add_argument('--model_name', type=str, default='answerdotai/ModernBERT-base', help='Name of the pre-trained model')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save the model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=512, help='Evaluation batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Number of warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer and create datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_df, dev_df, test_df = load_data(args.data_dir)
    train_dataset, eval_dataset, test_dataset = create_datasets(train_df, dev_df, test_df, tokenizer, model_type=args.model_type)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Initialize model and move to GPU
    model = ModernBertForSequenceClassification.from_pretrained(
        args.model_name,
        attn_implementation="flash_attention_2",
        num_labels=2
    )
    model = model.to(device)

    # Define training arguments with tensorboard logging
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1_class1",
        save_total_limit=3,
        report_to=["tensorboard"],
        logging_first_step=True,
        bf16=True,
        bf16_full_eval=True,
        push_to_hub=False,
    )

    # Initialize trainer with custom callbacks
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[
            PrinterCallback()
        ]
    )

    # Train the model
    train_result = trainer.train()
    
    # Print final metrics
    print("\nTraining completed:")
    print(f"Runtime: {train_result.metrics['train_runtime']:.2f}s")
    print(f"Samples/second: {train_result.metrics['train_samples_per_second']:.2f}")
    print(f"Final loss: {train_result.metrics['train_loss']:.4f}")

    # Save the final model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))

if __name__ == "__main__":
    main()
