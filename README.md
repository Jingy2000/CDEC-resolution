# Cross-Document Event Coreference Resolution

This project implements advanced transformer-based approaches for cross-document event coreference resolution. The model determines whether two event mentions from different documents refer to the same real-world event.

## Project Overview

- Developed and evaluated three transformer-based approaches:
  1. **Fine-tuned BERT Classification Model**: Using ModernBERT with a classification head to identify event coreference
  2. **Instruction-tuned Language Model**: Leveraging Qwen2.5 with specialized instructions for the coreference task
  3. **Reasoning-based Approach**: Using GRPO (Group Relative Policy Optimization) to train models for reasoning-based coreference resolution
- Comprehensive evaluation metrics and analysis
- Scalable pipeline for both training and inference

## Technical Implementation

### ModernBERT Classifier Approach
- Uses `ModernBERT-base` as foundation
- Event trigger pairs are highlighted with special tokens
- Binary classification head determines coreference relation

### Qwen Instruction Approach
- Leverages `Qwen2.5-1.5B-Instruct` capabilities
- Structures the task as an instruction-following problem
- Employs parameter-efficient fine-tuning (LoRA)

### Reasoning-based Approach
- Uses `Qwen2.5-1.5B-Instruct` as the base model
- Implements GRPO (Guided Reinforcement with Preference Optimization) training
- Explicitly generates reasoning steps before making coreference decisions
- Optimizes for multiple reward components including reasoning quality and final prediction accuracy

## Results

Results from our evaluation on the test dataset:

| Training Approach | Model | Precision | Recall | F1-score | Accuracy | Invalid Rate |
|-------------------|-------|-----------|--------|----------|----------|--------------|
| Direct Label SFT | ModernBERT | 0.7740 | 0.8316 | 0.8018 | 0.9632 | N/A |
| Direct Label SFT | Qwen2.5-1.5B-Instruct (SFT) | 0.8203 | 0.7936 | 0.8067 | 0.9660 | N/A |
| Reasoning-based | Qwen2.5-1.5B-Instruct (GRPO) | 0.7784 | 0.6385 | 0.7016 | 0.9511 | 3.04% |
| DeepSeek | deepseek_r1 | 0.9401 | 0.7822 | 0.8539 | 0.8705 | N/A |

The ModernBERT classifier and Qwen SFT models achieved the highest F1-scores among our trained models, with DeepSeek showing even better performance as an external reference. The reasoning-based approach with GRPO showed promising results with a low invalid output rate, demonstrating the potential of incorporating explicit reasoning steps.

## Getting Started

1. Install dependencies with poetry:
```bash
poetry install
```

2. Prepare your data in the `data` directory with the following structure:
```
data/
  ├── train_set.csv
  ├── dev_set.csv
  └── test_set.csv
  └── balanced_train_set.csv
```

### Data Format

The dataset should include pairs of sentences with event triggers and their coreference labels:
- `sentence1`: First document sentence containing event mention
- `sentence2`: Second document sentence containing event mention
- `e1_trigger`: Event trigger word in sentence1
- `e2_trigger`: Event trigger word in sentence2
- `label`: Binary coreference label (1 for coreferent, 0 for non-coreferent)


### Training

```bash
# Train ModernBERT model
python scripts/train_modernbert.py \
  --model_name answerdotai/ModernBERT-base \
  --data_dir data \
  --output_dir models/modernbert \
  --epochs 3 \
  --train_batch_size 64 \
  --learning_rate 1e-5
```

```bash
# Train Qwen model with LoRA (direct label SFT)
python scripts/train_qwen_instruct_sft.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --data_dir data \
  --output_dir models/qwen_instruct \
  --num_epochs 1 \
  --batch_size 4
```

```bash
# Train Qwen reasoning model with GRPO
python scripts/train_qwen_reason_grpo.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --data_dir data \
  --output_dir models/qwen_reason_grpo \
  --num_epochs 1 \
  --batch_size 4
```

### Evaluation

```bash
# Evaluate ModernBERT model
python scripts/eval_modernbert.py --model_path models/modernbert/checkpoint-1000-best --data_dir data

# Evaluate Qwen SFT model
python scripts/eval_qwen_instruct.py --base_model Qwen/Qwen2.5-1.5B-Instruct --adapter_path models/qwen_instruct/final --data_dir data

# Evaluate Qwen GRPO reasoning model
python scripts/eval_qwen_reason.py --base_model Qwen/Qwen2.5-1.5B-Instruct --adapter_path models/qwen_reason_grpo/final --data_dir data

# Analyze reasoning outputs
python scripts/eval_reasoning_results.py --results_file results/qwen_reason_grpo_results.jsonl
```

