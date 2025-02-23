import random
import os
import numpy as np
import torch
import pandas as pd

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data_to_df(data_dir):
    # Load datasets
    train_df = pd.read_csv(f"{data_dir}/train_set.csv")
    dev_df = pd.read_csv(f"{data_dir}/dev_set.csv")
    test_df = pd.read_csv(f"{data_dir}/test_set.csv")
    
    # oversample train set
    train_df = pd.concat([train_df[train_df['label'] == 1]] * 3 + [train_df[train_df['label'] == 0]])
    
    return train_df, dev_df, test_df