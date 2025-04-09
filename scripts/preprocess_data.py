"""
This script processes event coreference data through several stages:

1. Data Format Conversion:
   - Converts raw event pairs data files to CSV format
   - Handles train, dev, and test sets separately
   - Extracts trigger words and adds length information
   - Input files: event_pairs.{train|dev|test}
   - Output files: {train|dev|test}_set.csv

2. Balanced Dataset Creation:
   - Takes the full training set (~227k samples)
   - Keeps all positive samples (~19.6k)
   - Randomly samples negative examples (20k)
   - Total: ~39.6k balanced samples
   - Output: balanced_train_set.csv

3. Unique Sample Creation:
   - Takes the balanced dataset
   - Creates a smaller dataset with unique sentences
   - Samples 9k examples while maintaining diversity
   - Output: unique_sample_9k_reason.csv

Data Format Details:
- Each row represents a pair of event mentions
- Features include:
  * Sentences containing the events
  * Trigger word indices
  * Participant, time, and location spans (if available)
  * Binary label (1: coreferent, 0: non-coreferent)
- Test set also includes event IDs

Usage:
    python sample_train_data.py

The script will process all stages in sequence and save intermediate files.
Progress and statistics are printed for each stage.
"""

import pandas as pd

def create_balanced_dataset(input_file: str, num_negative_samples: int = 20000, random_state: int = 42):
    """
    Create a balanced dataset with all positive examples and a specified number of negative examples.
    
    Args:
        input_file: Path to the input CSV file
        num_negative_samples: Number of negative samples to include (default: 20000)
        random_state: Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Balanced dataset with all positive samples and sampled negative samples
    """
    train_df = pd.read_csv(input_file)
    
    # Split into positive and negative samples
    positive_samples = train_df[train_df['label'] == 1]
    negative_samples = train_df[train_df['label'] == 0]
    
    # Sample negative examples
    sampled_negative = negative_samples.sample(n=num_negative_samples, random_state=random_state)
    
    # Combine positive and negative samples
    balanced_dataset = pd.concat([positive_samples, sampled_negative], ignore_index=True)
    
    # Shuffle the dataset
    balanced_dataset = balanced_dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return balanced_dataset

def sample_unique_data(input_file: str, sample_size: int = 9000, random_state: int = 42):
    """
    Load and sample data from the input CSV file
    """
    train_df = pd.read_csv(input_file)
    
    # First ensure we're working with unique sentence1 and sentence2
    train_df = train_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    unique_s1 = train_df.drop_duplicates('sentence1')
    unique_s2 = train_df.drop_duplicates('sentence2')
    
    # Sample from each
    s1_sample = unique_s1.sample(min(sample_size, len(unique_s1)))
    s2_sample = unique_s2.sample(min(sample_size, len(unique_s2)))
    
    # Combine them and drop duplicates again if needed
    combined_sample = pd.concat([s1_sample, s2_sample]).drop_duplicates()
    
    # If we need exactly sample_size rows, we might need to sample more
    if len(combined_sample) < sample_size:
        # Get more rows from the original df that aren't in our sample yet
        remaining = train_df[~train_df.index.isin(combined_sample.index)]
        additional = remaining[remaining['label'] == 1].sample(min(sample_size - len(combined_sample), len(remaining)))
        final_sample = pd.concat([combined_sample, additional])
    else:
        # If we have more than sample_size, just take sample_size
        final_sample = combined_sample.sample(sample_size)
    
    final_sample = final_sample.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return final_sample

def print_dataset_stats(df: pd.DataFrame):
    """
    Print statistics about the dataset
    """
    print(f"Unique sentence1: {df['sentence1'].nunique()}")
    print(f"Unique sentence2: {df['sentence2'].nunique()}")
    print(f"Unique sentences overall: {pd.concat([df.sentence1, df.sentence2]).nunique()}")
    print("\nLabel distribution:")
    print(df.label.value_counts())

def save_dataset(df: pd.DataFrame, output_file: str):
    """
    Save the dataset to a CSV file
    """
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")

def convert_event_pairs_to_csv(input_file: str, has_event_ids: bool = False) -> pd.DataFrame:
    """
    Convert event pairs data file to a pandas DataFrame and save as CSV.
    
    Args:
        input_file: Path to the event pairs data file
        has_event_ids: Whether the input file contains event IDs (True for test set)
        
    Returns:
        pd.DataFrame: Processed DataFrame with event pairs data
    """
    # Define column names based on the data format
    col_names = [
        "sentence1",
        "e1_trigger_start",
        "e1_trigger_end",
        "e1_participant1_start",
        "e1_participant1_end",
        "e1_participant2_start",
        "e1_participant2_end",
        "e1_time_start",
        "e1_time_end",
        "e1_loc_start",
        "e1_loc_end",
        "sentence2",
        "e2_trigger_start",
        "e2_trigger_end",
        "e2_participant1_start",
        "e2_participant1_end",
        "e2_participant2_start",
        "e2_participant2_end",
        "e2_time_start",
        "e2_time_end",
        "e2_loc_start",
        "e2_loc_end",
        "label"
    ]
    
    # Read the data file
    with open(input_file) as f:
        lines = f.readlines()
        lines = [l.strip().split("\t") for l in lines]
    
    # Create DataFrame
    if has_event_ids:
        # For test set, which includes event IDs
        df = pd.DataFrame(lines, columns=['event_id_1', 'event_id_2'] + col_names)
    else:
        # For train and dev sets
        df = pd.DataFrame(lines, columns=col_names)
    
    # Convert numeric columns to integers
    numeric_cols = [
        'e1_trigger_start', 'e1_trigger_end',
        'e2_trigger_start', 'e2_trigger_end',
        'label'
    ]
    
    df[numeric_cols] = df[numeric_cols].astype(int)
    
    # Convert text columns to string
    text_cols = ['sentence1', 'sentence2']
    df[text_cols] = df[text_cols].astype(str)
    
    # Extract trigger words
    df['e1_trigger'] = df.apply(lambda row: extract_trigger(row, 1), axis=1)
    df['e2_trigger'] = df.apply(lambda row: extract_trigger(row, 2), axis=1)
    
    # Calculate total length of sentences
    df['length'] = df.apply(lambda row: len(row['sentence1'].split()) + len(row['sentence2'].split()), axis=1)
    
    return df

def extract_trigger(row, i: str):
    """
    Extract trigger word from sentence based on trigger indices.
    
    Args:
        row: DataFrame row
        i: Event number (1 or 2)
        
    Returns:
        str: Extracted trigger word(s) or None if extraction fails
    """
    try:
        trigger_start = row[f'e{i}_trigger_start']
        trigger_end = row[f'e{i}_trigger_end']
        
        # Check for NaN values
        if pd.isna(trigger_start) or pd.isna(trigger_end):
            return None
            
        start_idx = int(trigger_start)
        end_idx = int(trigger_end)
        
        sentence = row[f'sentence{i}']
            
        words = sentence.split()
        if start_idx >= 0 and end_idx < len(words):
            return " ".join(words[start_idx: end_idx + 1])
        else:
            raise ValueError("Trigger indices out of bounds")
        
    except Exception as e:
        print(f"Error in row {row.name}: {str(e)}")
        return None

if __name__ == "__main__":
    # First convert event pairs data to CSV format
    print("Converting event pairs data to CSV format...")
    
    # Convert train set
    train_pairs_file = "../data/event_pairs.train"
    train_csv_file = "../data/train_set.csv"
    train_df = convert_event_pairs_to_csv(train_pairs_file)
    save_dataset(train_df, train_csv_file)
    print("\nTrain set statistics:")
    print_dataset_stats(train_df)
    
    # Convert dev set
    dev_pairs_file = "../data/event_pairs.dev"
    dev_csv_file = "../data/dev_set.csv"
    dev_df = convert_event_pairs_to_csv(dev_pairs_file)
    save_dataset(dev_df, dev_csv_file)
    print("\nDev set statistics:")
    print_dataset_stats(dev_df)
    
    # Convert test set
    test_pairs_file = "../data/event_pairs.test"
    test_csv_file = "../data/test_set.csv"
    test_df = convert_event_pairs_to_csv(test_pairs_file, has_event_ids=True)
    save_dataset(test_df, test_csv_file)
    print("\nTest set statistics:")
    print_dataset_stats(test_df)
    
    # Create balanced dataset
    print("\nCreating balanced dataset...")
    balanced_output = "../data/balanced_train_set.csv"
    balanced_df = create_balanced_dataset(train_csv_file)
    print("\nBalanced dataset statistics:")
    print_dataset_stats(balanced_df)
    save_dataset(balanced_df, balanced_output)
    
    # Create unique sampled dataset
    print("\nCreating unique sampled dataset...")
    unique_sample_output = "../data/unique_sample_9k_reason.csv"
    final_sample = sample_unique_data(balanced_output)
    print("\nUnique sampled dataset statistics:")
    print_dataset_stats(final_sample)
    save_dataset(final_sample, unique_sample_output) 