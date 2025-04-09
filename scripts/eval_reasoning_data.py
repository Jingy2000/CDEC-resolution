import pandas as pd
from typing import List, Dict, Tuple

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
        'accuracy': micro_accuracy
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

def print_metrics(metrics: Dict) -> None:
    """
    Print evaluation metrics in a formatted way
    """
    print("Per-class metrics:")
    for cls in metrics['per_class']:
        print(f"\nClass {cls}:")
        print(f"Precision: {metrics['per_class'][cls]['precision']:.3f}")
        print(f"Recall: {metrics['per_class'][cls]['recall']:.3f}")
        print(f"F1: {metrics['per_class'][cls]['f1']:.3f}")
    
    print("\nMacro average:")
    print(f"Precision: {metrics['macro_avg']['precision']:.3f}")
    print(f"Recall: {metrics['macro_avg']['recall']:.3f}")
    print(f"F1: {metrics['macro_avg']['f1']:.3f}")
    print(f"\nOverall Accuracy: {metrics['accuracy']:.3f}")

if __name__ == "__main__":
    # Load results
    results_df = pd.read_csv("../data/results_final.csv")
    
    # Process responses
    true_labels, pred_labels = process_model_responses(results_df)
    
    # Calculate and print metrics
    metrics = calculate_multi_class_metrics(true_labels, pred_labels)
    print_metrics(metrics)
    
    # Save processed results
    reason_df = results_df[pd.notna(results_df['reasoning_content'])]
    reason_df.to_csv("../data/reason_deepseek_r1_train.csv", index=False) 