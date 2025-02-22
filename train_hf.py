import os
import torch
import argparse
from transformers import (
    AutoTokenizer, 
    ModernBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from src.data import load_data, create_datasets
from src.utils import set_seed
import evaluate
import numpy as np
import wandb
from src.callbacks import CustomWandbCallback, PrinterCallback

def parse_args():
    parser = argparse.ArgumentParser(description='Train CDEC Resolution Model with HuggingFace Trainer')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the data files')
    parser.add_argument('--model_name', type=str, default='answerdotai/ModernBERT-base', help='Name of the pre-trained model')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save the model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=128, help='Evaluation batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Number of warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--wandb_project', type=str, default='cdec-resolution', help='Weights & Biases project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Weights & Biases run name')
    return parser.parse_args()

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')
    precision = precision_metric.compute(predictions=predictions, references=labels, average='weighted')
    recall = recall_metric.compute(predictions=predictions, references=labels, average='weighted')
    
    return {
        'accuracy': accuracy['accuracy'],
        'f1': f1['f1'],
        'precision': precision['precision'],
        'recall': recall['recall']
    }

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args)
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer and create datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_df, dev_df, test_df = load_data(args.data_dir)
    train_dataset, eval_dataset, test_dataset = create_datasets(train_df, dev_df, test_df, tokenizer)

    # Initialize model
    model = ModernBertForSequenceClassification.from_pretrained(
        args.model_name,
        attn_implementation="flash_attention_2",
        num_labels=2
    )

    # Define training arguments with additional logging
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        evaluation_strategy="steps",  # Changed to steps for more frequent eval
        eval_steps=100,  # Evaluate every 100 steps
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,  # Log every 10 steps
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        report_to=["wandb", "tensorboard"],
        # Additional arguments for better logging
        logging_first_step=True,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        dataloader_num_workers=4,
        group_by_length=True,  # Reduces padding, speeds up training
        fp16=True  # Use mixed precision training
    )

    # Initialize trainer with custom callbacks
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            CustomWandbCallback(),
            PrinterCallback()
        ]
    )

    # Train the model
    train_result = trainer.train()
    
    # Log final metrics
    wandb.log({
        "train_runtime": train_result.metrics["train_runtime"],
        "train_samples_per_second": train_result.metrics["train_samples_per_second"],
        "train_loss": train_result.metrics["train_loss"],
    })

    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset, metric_key_prefix="test")
    print("\nTest Results:", test_results)
    wandb.log({"test_" + k: v for k, v in test_results.items()})

    # Save the final model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
