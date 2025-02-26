# CDEC-resolution

Cross-Document Event Coreference Resolution using Transformer Models.

## Setup

1. Install dependencies:
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

Each CSV file should contain columns: `sentence1`, `sentence2`, `e1_trigger`, `e2_trigger`, and `label`.

## Training

### Training BERT Model

```bash
python scripts/train_bert.py \
  --model_name answerdotai/ModernBERT-base \
  --data_dir data \
  --output_dir models/bert \
  --epochs 3 \
  --train_batch_size 64 \
  --learning_rate 1e-5
```

### Training Qwen Model (SFT)

```bash
python scripts/train_sft.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --data_dir data \
  --output_dir models/qwen \
  --num_epochs 1 \
  --batch_size 4
```

## Evaluation

### Evaluating BERT Model

```bash
python scripts/eval_bert.py \
  --model_path models/bert/final_model \
  --data_dir data \
  --output_dir evaluation_results
```

### Evaluating Qwen Model

```bash
python scripts/eval_sft.py \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_path models/qwen/final \
  --data_dir data \
  --output_dir evaluation_results
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
