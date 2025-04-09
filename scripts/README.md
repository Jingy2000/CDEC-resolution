# Data Processing Scripts

This directory contains scripts for processing the event coreference dataset.

## Scripts Overview

### `sample_train_data.py`

A comprehensive data processing script that handles the complete pipeline from raw event pairs data to the final sampled dataset.

#### Pipeline Stages

```mermaid
graph TD
    A[Raw Event Pairs Data] --> B[CSV Conversion]
    B --> C[Balanced Dataset]
    C --> D[Unique Sampled Dataset]
    
    subgraph "Stage 1: CSV Conversion"
    B1[event_pairs.train] --> B2[train_set.csv]
    B3[event_pairs.dev] --> B4[dev_set.csv]
    B5[event_pairs.test] --> B6[test_set.csv]
    end
    
    subgraph "Stage 2: Balance Dataset"
    C1[Full Training Set] --> C2[All Positive Samples]
    C1 --> C3[Sampled Negative Examples]
    C2 --> C4[Balanced Dataset]
    C3 --> C4
    end
    
    subgraph "Stage 3: Create Sample"
    D1[Balanced Dataset] --> D2[Sample Unique Sentences]
    D2 --> D3[Final 9k Dataset]
    end
```

#### Usage

```bash
python sample_train_data.py
```

#### Output Files

- `train_set.csv`: Converted training data
- `dev_set.csv`: Converted development data
- `test_set.csv`: Converted test data
- `balanced_train_set.csv`: Balanced dataset
- `unique_sample_9k_reason.csv`: Final sampled dataset

#### Data Format

Each row in the CSV files contains:
- Sentence pairs with event mentions
- Trigger word indices and extracted triggers
- Participant, time, and location spans (if available)
- Binary coreference label (1: coreferent, 0: non-coreferent)
- Total sentence length
- Event IDs (test set only) 

### `reasoning_collector.py`

A script for collecting reasoning data from the DeepSeek API for event coreference analysis. Used `asyncio` for efficient batch processing.

#### Usage

```bash
python reasoning_collector.py
```

Required variables:
- `api_key`: Your DeepSeek API key, you need to modify the script to run

#### Output Files

- `results_intermediate.csv`: Saves progress during processing
- `results_final.csv`: Complete dataset with model responses

### `eval_reasoning_data.py`

A comprehensive evaluation script for analyzing model performance on event coreference predictions.

#### Key Features

1. **Metrics Calculation**
   - Precision, Recall, and F1 score
   - Per-class metrics
   - Macro and micro averages
   - Support statistics

2. **Multi-class Evaluation**
   - Handles binary and multi-class scenarios
   - Detailed per-class performance analysis
   - Aggregate statistics across classes

3. **Results Processing**
   - Converts model responses to binary labels
   - Handles missing or invalid responses
   - Generates formatted evaluation reports

#### Usage

```bash
python eval_reasoning_data.py
```

#### Output Files

- `reason_deepseek_r1_train.csv`: Processed results with reasoning content

#### Metrics Output Format

```
Per-class metrics:
Class 0:
  Precision: X.XXX
  Recall: X.XXX
  F1: X.XXX

Class 1:
  Precision: X.XXX
  Recall: X.XXX
  F1: X.XXX

Macro average:
  Precision: X.XXX
  Recall: X.XXX
  F1: X.XXX

Overall Accuracy: X.XXX
``` 