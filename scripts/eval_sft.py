import argparse
import torch
import json
import os
from tqdm import tqdm
from datetime import datetime
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix
from src.utils import load_data_to_df
from src.data_sft import generate_coreference_message

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CDEC SFT Model')
    parser.add_argument('--base_model', type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                      help='Base model name')
    parser.add_argument('--adapter_path', type=str, required=True,
                      help='Path to the LoRA adapter')
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Directory containing the test data')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size for evaluation')
    return parser.parse_args()

def evaluate_predictions(true_labels, predicted_labels):
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=['Not Coreferent', 'Coreferent'],
        output_dict=True
    )
    
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Calculate additional metrics
    true_positives = cm[1][1]
    false_positives = cm[0][1]
    true_negatives = cm[0][0]
    false_negatives = cm[1][0]
    
    return {
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'additional_metrics': {
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'true_negatives': int(true_negatives),
            'false_negatives': int(false_negatives)
        }
    }

def process_model_output(output):
    # Convert model output to binary prediction
    output = output.strip().lower()
    return 1 if output == 'yes' else 0

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer
    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    print(f"Loading adapter from: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    
    # Load test data
    print("Loading test data...")
    _, _, test_df = load_data_to_df(args.data_dir)
    
    # sample for testing
    test_df = test_df.sample(frac=0.01)
    
    # Generate predictions
    print("Generating predictions...")
    predictions = []
    true_labels = test_df['label'].tolist()
    
    model.eval()
    with torch.no_grad():
        # Process data in batches
        for i in tqdm(range(0, len(test_df), args.batch_size)):
            batch_df = test_df.iloc[i:i + args.batch_size]
            batch_messages = [generate_coreference_message(row) for _, row in batch_df.iterrows()]
            batch_prompts = [tokenizer.apply_chat_template(msgs[:-1], tokenize=False) 
                           for msgs in batch_messages]
            
            # Tokenize all prompts in the batch
            batch_inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                padding_side="left",
                truncation=True
            ).to(device)
            
            # Generate outputs for the entire batch
            outputs = model.generate(
                **batch_inputs,
                max_new_tokens=5,
                temperature=1,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Decode all outputs in the batch
            responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Process each response in the batch
            for response in responses:
                answer = response.split("Answer:")[-1].strip()
                prediction = process_model_output(answer)
                predictions.append(prediction)
    
    # Calculate metrics
    results = evaluate_predictions(true_labels, predictions)
    results['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results['model_info'] = {
        'base_model': args.base_model,
        'adapter_path': args.adapter_path
    }
    
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
    print("\nMetrics for Coreferent class:")
    print(f"Precision: {results['classification_report']['Coreferent']['precision']:.3f}")
    print(f"Recall: {results['classification_report']['Coreferent']['recall']:.3f}")
    print(f"F1-Score: {results['classification_report']['Coreferent']['f1-score']:.3f}")
    print(f"\nResults saved to: {results_file}")
    print("=" * 30)

if __name__ == "__main__":
    main()
