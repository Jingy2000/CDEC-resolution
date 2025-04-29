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
- Implements GRPO (Group Relative Policy Optimization) training
- Explicitly generates reasoning steps before making coreference decisions
- Optimizes for multiple reward components including reasoning quality and final prediction accuracy

## Results

Results from our evaluation on the test dataset:

### Primary Models

| Training Approach | Model | Precision | Recall | F1-score | Accuracy | Invalid Rate |
|-------------------|-------|-----------|--------|----------|----------|--------------|
| Direct Label SFT | ModernBERT | 0.7740 | 0.8316 | 0.8018 | 0.9632 | N/A |
| Direct Label SFT | Qwen2.5-1.5B-Instruct (SFT) | 0.8203 | 0.7936 | 0.8067 | 0.9660 | N/A |
| Reasoning-based | Qwen2.5-1.5B-Instruct (GRPO) | 0.7784 | 0.6385 | 0.7016 | 0.9511 | 3.04% |
| External | DeepSeek-R1 | 0.9401 | 0.7822 | 0.8539 | 0.8705 | N/A |

### Baselines and Additional Models

| Category | Model | Precision | Recall | F1-score | Accuracy | Invalid Rate |
|----------|-------|-----------|--------|----------|----------|--------------|
| Baseline | Qwen2.5-1.5B-Instruct (baseline) | 0.8850 | 0.4745 | 0.6178 | 0.9475 | N/A |
| Baseline | Qwen2.5-1.5B-Instruct-reason-baseline | 0.8214 | 0.3485 | 0.4894 | 0.9389 | 26.87% |
| Alternative | DeepSeek-R1-Distill-Qwen-1.5B | 0.4661 | 0.4755 | 0.4708 | 0.9044 | N/A |
| Alternative | Qwen2.5-1.5B-Instruct (SFT on reasoning) | 0.5501 | 0.3361 | 0.4173 | 0.9169 | 1.69% |

The ModernBERT classifier and Qwen SFT models achieved the highest F1-scores among our trained models, with DeepSeek showing even better performance as an external reference. The reasoning-based approach with GRPO showed promising results with a low invalid output rate, demonstrating the potential of incorporating explicit reasoning steps.

Notably, our fine-tuned models substantially outperformed their respective baselines, with the SFT approach improving Qwen's F1-score from 0.6178 to 0.8067, and GRPO improving the reasoning baseline from 0.4894 to 0.7016 while drastically reducing invalid outputs from 26.87% to just 3.04%.

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

