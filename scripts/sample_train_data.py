import pandas as pd
import numpy as np

def load_and_sample_data(input_file: str, sample_size: int = 9000, random_state: int = 42):
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

if __name__ == "__main__":
    # Example usage
    input_file = "../data/balanced_train_set.csv"
    output_file = "../data/unique_sample_9k_reason.csv"
    
    # Load and sample data
    final_sample = load_and_sample_data(input_file)
    
    # Print statistics
    print_dataset_stats(final_sample)
    
    # Save the dataset
    save_dataset(final_sample, output_file) 