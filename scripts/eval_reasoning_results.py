import pandas as pd
from typing import List, Dict, Tuple
import json
import datetime
import numpy as np
import os
from sklearn.metrics import confusion_matrix

def calculate_metrics(true_labels: List[int], pred_labels: List[int]) -> Dict[str, float]:
    """
    Calculate precision, recall, F1, and accuracy
    
    Args:
        true_labels: list of true labels
        pred_labels: list of predicted labels
    """
    # Initialize counters
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    
    # Count all cases
    for true, pred in zip(true_labels, pred_labels):
        if true == 1 and pred == 1:
            true_positive += 1
        elif true == 0 and pred == 0:
            true_negative += 1
        elif true == 0 and pred == 1:
            false_positive += 1
        elif true == 1 and pred == 0:
            false_negative += 1
    
    # Calculate metrics
    epsilon = 1e-10  # Small value to prevent division by zero
    
    precision = true_positive / (true_positive + false_positive + epsilon)
    recall = true_positive / (true_positive + false_negative + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'support': {
            'true_positive': true_positive,
            'true_negative': true_negative,
            'false_positive': false_positive,
            'false_negative': false_negative
        }
    }

def calculate_multi_class_metrics(true_labels: List[int], pred_labels: List[int]) -> Dict[str, Dict]:
    """
    Calculate metrics for multiple classes
    
    Args:
        true_labels: list of true labels
        pred_labels: list of predicted labels
    """
    # Get unique classes
    classes = sorted(list(set(true_labels + pred_labels)))
    
    # Initialize results dictionary
    results = {}
    
    # Calculate metrics for each class
    for cls in classes:
        # Convert to binary classification problem
        true_binary = [1 if label == cls else 0 for label in true_labels]
        pred_binary = [1 if label == cls else 0 for label in pred_labels]
        
        # Calculate metrics for this class
        metrics = calculate_metrics(true_binary, pred_binary)
        results[cls] = metrics
    
    # Calculate macro average
    macro_precision = sum(results[cls]['precision'] for cls in classes) / len(classes)
    macro_recall = sum(results[cls]['recall'] for cls in classes) / len(classes)
    macro_f1 = sum(results[cls]['f1'] for cls in classes) / len(classes)
    
    # Calculate micro average (overall accuracy)
    total_correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    micro_accuracy = total_correct / len(true_labels)
    
    return {
        'per_class': results,
        'macro_avg': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        },
        'accuracy': micro_accuracy,
        
    }

def process_model_responses(df: pd.DataFrame) -> Tuple[List[int], List[int]]:
    """
    Process model responses and convert them to binary labels
    """
    def is_coreference(content: str) -> int:
        return 1 if "yes" in content.strip().lower() else 0
    
    # Filter out rows without responses
    df_with_responses = df[df['model_response'].notna()]
    
    # Get true labels and predictions
    true_labels = df_with_responses.label.tolist()
    pred_labels = [is_coreference(pred) for pred in df_with_responses.model_response]
    
    return true_labels, pred_labels

def format_results_as_json(true_labels: List[int], pred_labels: List[int]) -> Dict:
    """
    Format results as JSON in the desired format
    """
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, pred_labels).tolist()
    
    # Get total support counts
    total_samples = len(true_labels)
    class_0_samples = true_labels.count(0)
    class_1_samples = true_labels.count(1)
    
    # Calculate metrics for each class
    # For class 0 (Not Coreferent)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    
    # Not Coreferent metrics
    not_coref_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    not_coref_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    not_coref_f1 = 2 * (not_coref_precision * not_coref_recall) / (not_coref_precision + not_coref_recall) if (not_coref_precision + not_coref_recall) > 0 else 0
    
    # Coreferent metrics
    coref_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    coref_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    coref_f1 = 2 * (coref_precision * coref_recall) / (coref_precision + coref_recall) if (coref_precision + coref_recall) > 0 else 0
    
    # Overall accuracy
    accuracy = (tp + tn) / total_samples
    
    # Macro averages
    macro_precision = (not_coref_precision + coref_precision) / 2
    macro_recall = (not_coref_recall + coref_recall) / 2
    macro_f1 = (not_coref_f1 + coref_f1) / 2
    
    # Weighted averages
    weighted_precision = (not_coref_precision * class_0_samples + coref_precision * class_1_samples) / total_samples
    weighted_recall = (not_coref_recall * class_0_samples + coref_recall * class_1_samples) / total_samples
    weighted_f1 = (not_coref_f1 * class_0_samples + coref_f1 * class_1_samples) / total_samples
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        "classification_report": {
            "Not Coreferent": {
                "precision": not_coref_precision,
                "recall": not_coref_recall,
                "f1-score": not_coref_f1,
                "support": float(class_0_samples)
            },
            "Coreferent": {
                "precision": coref_precision,
                "recall": coref_recall,
                "f1-score": coref_f1,
                "support": float(class_1_samples)
            },
            "accuracy": accuracy,
            "macro avg": {
                "precision": macro_precision,
                "recall": macro_recall,
                "f1-score": macro_f1,
                "support": float(total_samples)
            },
            "weighted avg": {
                "precision": weighted_precision,
                "recall": weighted_recall,
                "f1-score": weighted_f1,
                "support": float(total_samples)
            }
        },
        "confusion_matrix": cm,
        "additional_metrics": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn
        },
        "timestamp": timestamp,
    }

def save_results(results: Dict, output_path: str = None) -> None:
    """
    Save results to a JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # Load results
    results_df = pd.read_csv("data/reason_deepseek_r1_all.csv")
    
    # Process responses
    true_labels, pred_labels = process_model_responses(results_df)
    
    # Format results as JSON
    results_json = format_results_as_json(true_labels, pred_labels)
    
    # Print results
    print(json.dumps(results_json, indent=4))
    
    # Save results to file
    save_results(results_json, output_path="evaluation_results/deepseek_r1.json")
    
    # Save processed results if needed
    reason_df = results_df[pd.notna(results_df['reasoning_content'])]