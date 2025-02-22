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
    return parser.parse_args()

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')
    
    return {
        'accuracy': accuracy['accuracy'],
        'f1': f1['f1']
    }

def main():
    args = parse_args()
    set_seed(args.seed)
    
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

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        report_to="tensorboard"
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train the model
    trainer.train()

    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)
    print("\nTest Results:", test_results)

    # Save the final model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))

if __name__ == "__main__":
    main()
