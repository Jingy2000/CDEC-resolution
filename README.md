# Cross-Document Event Coreference Resolution

This project implements advanced transformer-based approaches for cross-document event coreference resolution. The model determines whether two event mentions from different documents refer to the same real-world event.

## Project Overview

- Developed and evaluated two transformer-based approaches:
  1. **Fine-tuned BERT Classification Model**: Using BERT with a classification head to identify event coreference
  2. **Instruction-tuned Language Model**: Leveraging Qwen2.5 with specialized instructions for the coreference task
- Comprehensive evaluation metrics and analysis
- Scalable pipeline for both training and inference

## Technical Implementation

### ModernBERT Classifier Approach
- Uses `ModernBERT-base` as foundation
- Event trigger pairs are highlighted with special tokens
- Binary classification head determines coreference relation

### Qwen Instruction Approach
- Leverages `Qwen2.5-0.5B-Instruct` capabilities
- Structures the task as an instruction-following problem
- Employs parameter-efficient fine-tuning (LoRA)

## Results
   

| Model | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| ModernBERT Classifier | 0.858 | 0.654 | 0.742 |
| Qwen2.5-0.5B-Instruct | 0.828 | 0.464 | 0.594 |

The encoder-based model achieved superior performance, demonstrating that ModernBERT is better at detecting actual event coreference relationships, whereas Qwen tends to miss many coreferent pairs. One possible reason for this difference is that ModernBERT is encoder based model and trained on a lot of language understanding tasks, meaning it can fully leverage contextual information from both sentences. In contrast, Qwen is a decoder-only model, which might lead it to focus more on typical language patterns rather than deeply understanding event semantics.


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
# Train BERT model
python scripts/train_bert.py \
  --model_name answerdotai/ModernBERT-base \
  --data_dir data \
  --output_dir models/bert \
  --epochs 3 \
  --train_batch_size 64 \
  --learning_rate 1e-5
```

```bash
# Train Qwen model with LoRA
python scripts/train_sft.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --data_dir data \
  --output_dir models/qwen \
  --num_epochs 1 \
  --batch_size 4
```

### Evaluation

```bash
# Evaluate BERT model
python scripts/eval_bert.py --model_path models/bert/final_model --data_dir data

# Evaluate Qwen model
python scripts/eval_sft.py --base_model Qwen/Qwen2.5-0.5B-Instruct --adapter_path models/qwen/final --data_dir data
```


## Project Structure

```
.
├── data/                   # Data directory
├── models/                 # Saved models directory
├── scripts/               
│   ├── train_bert.py      # Training script for BERT
│   ├── train_sft.py       # Training script for Qwen
│   ├── eval_bert.py       # Evaluation script for BERT
│   └── eval_sft.py        # Evaluation script for Qwen
└── src/
    ├── data_bert.py       # BERT data processing
    ├── data_sft.py        # Qwen data processing
    └── utils.py           # Utility functions
```


TODO:
1. SFT on reasoning data.
  1. 
