import argparse
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
from src.data_bert import create_single_dataloader
from src.utils import load_data_to_df
import json
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

torch._dynamo.config.suppress_errors = True

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CDEC Resolution Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the test data')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for evaluation')
    return parser.parse_args()

def evaluate_model(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(device)
            
            outputs = model(**inputs)
            logits = outputs.logits
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probs)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = model.to(device)
    
    # Load test data
    _, _, test_df = load_data_to_df(args.data_dir)
    test_loader = create_single_dataloader(
        test_df, 
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    if test_loader is None:
        raise ValueError("Failed to create test dataloader. Check if test data exists.")
    
    # Get predictions
    predictions, labels, probabilities = evaluate_model(model, test_loader, device)
    
    # Compute metrics
    report = classification_report(
        labels, 
        predictions,
        output_dict=True,
        target_names=['Not Coreferent', 'Coreferent']
    )
    
    cm = confusion_matrix(labels, predictions)
    
    # Calculate additional metrics
    true_positives = cm[1][1]
    false_positives = cm[0][1]
    true_negatives = cm[0][0]
    false_negatives = cm[1][0]
        
    # Prepare results
    results = {
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'additional_metrics': {
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'true_negatives': int(true_negatives),
            'false_negatives': int(false_negatives)
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_path': args.model_path
    }
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f'evaluation_results_{timestamp}.json')
    cm_plot_file = os.path.join(args.output_dir, f'confusion_matrix_{timestamp}.png')
    
    # Save results to JSON
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    print("\n=== Evaluation Results ===")
    print(f"Model path: {args.model_path}")
    print("\nMetrics for Coreferent class (label 1):")
    print(f"classification_report:\n{report}")
    print(f"\nResults saved to: {results_file}")
    print("=" * 30)

if __name__ == "__main__":
    main()
