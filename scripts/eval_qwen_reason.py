import argparse
import torch
import json
import os
import re
from tqdm import tqdm
from datetime import datetime
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix
from src.utils import load_data_to_df
from src.data_qwen_instruct import generate_coreference_message_qwen_reason

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CDEC SFT Model with Reasoning')
    parser.add_argument('--base_model', type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                      help='Base model name')
    parser.add_argument('--adapter_path', type=str, default=None,
                      help='Path to the LoRA adapter')
    parser.add_argument('--eval_path', type=str,default='data/test_set.csv',
                      help='Path to the evaluation dataset CSV file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Directory to save evaluation results')
    parser.add_argument('--tag', type=str, default='grpo',
                      help='Tag')
    return parser.parse_args()

def evaluate_predictions(true_labels, predicted_labels):
    # Filter out invalid predictions for classification report
    valid_indices = [i for i, pred in enumerate(predicted_labels) if pred != "Invalid"]
    filtered_true = [true_labels[i] for i in valid_indices]
    filtered_pred = [predicted_labels[i] for i in valid_indices]
    
    # Calculate invalid rate
    invalid_count = sum(1 for pred in predicted_labels if pred == "Invalid")
    invalid_rate = invalid_count / len(predicted_labels) if predicted_labels else 0
    
    # Generate classification report only on valid predictions
    report = classification_report(
        filtered_true,
        filtered_pred,
        target_names=['Not Coreferent', 'Coreferent'],
        output_dict=True
    ) if filtered_pred else {"No valid predictions": True}
    
    # Confusion matrix only on valid predictions
    cm = confusion_matrix(filtered_true, filtered_pred) if filtered_pred else np.zeros((2, 2))
    
    # Calculate additional metrics
    if len(filtered_pred) > 0:
        true_positives = cm[1][1]
        false_positives = cm[0][1]
        true_negatives = cm[0][0]
        false_negatives = cm[1][0]
    else:
        true_positives = false_positives = true_negatives = false_negatives = 0
    
    return {
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'additional_metrics': {
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'true_negatives': int(true_negatives),
            'false_negatives': int(false_negatives),
            'invalid_count': invalid_count,
            'invalid_rate': invalid_rate,
            'total_samples': len(predicted_labels)
        }
    }

def process_model_output(output):
    # Convert model output to binary prediction
    output = output.strip().lower()
    if 'yes' in output:
        return 1
    if 'no' in output:
        return 0
    else:
        return "Invalid"


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True
    )
    
    # # Load LoRA adapter
    # if args.adapter_path is not None:
    #     print(f"Loading adapter from: {args.adapter_path}")
    #     model = PeftModel.from_pretrained(model, args.adapter_path)
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(args.eval_path)
    
    # sample for testing
    # test_df = test_df.sample(frac=0.01)
    
    # Generate predictions
    print("Generating predictions...")
    true_labels = test_df['label'].tolist()
    
    messages = test_df.apply(lambda row: generate_coreference_message_qwen_reason(row)[:-1], axis=1)
    prompts = [tokenizer.apply_chat_template(
        msg, 
        tokenize=False,
        add_generation_prompt=True
    ) for msg in messages]

    
    # Load model and tokenizer
    print(f"Loading base model: {args.base_model}")
    
    llm = LLM(
        model=args.base_model,
        enable_lora=True if args.adapter_path is not None else False,
        max_lora_rank=32
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=1024
    )

    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest("trained_lora", 1, lora_path=args.adapter_path) if args.adapter_path is not None else None,
    )
    
    predictions = []
    pattern = r'<answer>(.*?)</answer>'
    
    for output in outputs:
        generated_text = output.outputs[0].text
        if len(generated_text.split('<answer>')) >= 2:
            answer = generated_text.split('<answer>')[1]
            prediction = process_model_output(answer)
        else:
            prediction = "Invalid"
        
        predictions.append(prediction)
    
    # Calculate metrics
    results = evaluate_predictions(true_labels, predictions)
    results['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results['model_info'] = {
        'base_model': args.base_model,
        'adapter_path': args.adapter_path
    }
    
    results['tag'] = args.tag
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f'sft_evaluation_results_{timestamp}.json')
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    print("\n=== Evaluation Results ===")
    print(f"Model: {args.base_model}")
    print(f"Adapter: {args.adapter_path}")
    print(f"\nInvalid predictions: {results['additional_metrics']['invalid_count']} ({results['additional_metrics']['invalid_rate']:.2%})")
    
    if "No valid predictions" not in results['classification_report']:
        print("\nMetrics for Coreferent class:")
        print(f"Precision: {results['classification_report']['Coreferent']['precision']:.3f}")
        print(f"Recall: {results['classification_report']['Coreferent']['recall']:.3f}")
        print(f"F1-Score: {results['classification_report']['Coreferent']['f1-score']:.3f}")
    else:
        print("\nNo valid predictions to calculate metrics")
    
    print(f"\nResults saved to: {results_file}")
    print("=" * 30)

if __name__ == "__main__":
    main()
