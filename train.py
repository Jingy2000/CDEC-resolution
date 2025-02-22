import torch
import argparse
from transformers import AutoTokenizer, ModernBertForSequenceClassification
from src.data import load_data, create_data_loaders
from src.trainer import Trainer
from src.utils import set_seed
from sklearn.metrics import classification_report

def parse_args():
    parser = argparse.ArgumentParser(description='Train CDEC Resolution Model')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the data files')
    parser.add_argument('--model_name', type=str, default='answerdotai/ModernBERT-base', help='Name of the pre-trained model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=128, help='Evaluation batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save the model')
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Load tokenizer and create data loaders
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_df, dev_df, test_df = load_data(args.data_dir)
    train_loader, dev_loader, test_loader = create_data_loaders(
        train_df, dev_df, test_df, tokenizer, 
        args.train_batch_size, args.eval_batch_size
    )

    # Initialize model
    model = ModernBertForSequenceClassification.from_pretrained(
        args.model_name,
        attn_implementation="flash_attention_2"
    ).to('cuda')

    # Initialize optimizer and trainer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    trainer = Trainer(model, optimizer)

    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = trainer.train_epoch(train_loader)
        dev_loss = trainer.evaluate(dev_loader)
        
        trainer.tracker.update({
            'train_loss': train_loss,
            'dev_loss': dev_loss
        })
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Dev Loss: {dev_loss:.4f}")
        
        # Save best model
        if dev_loss < best_loss:
            best_loss = dev_loss
            torch.save(model.state_dict(), f"{args.output_dir}/best_model.pt")

    # Plot training history
    trainer.tracker.plot()

    # Evaluate on test set
    predictions, actual_labels = trainer.predict(test_loader)
    print("\nTest Set Classification Report:")
    print(classification_report(actual_labels, predictions))

if __name__ == "__main__":
    main()
