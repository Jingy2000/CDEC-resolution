import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer

class CDECDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_len=512):
        """
        Args:
            df: DataFrame containing the data
            tokenizer: BERT tokenizer
            max_len (int): Maximum length of tokens
        """
        self.data = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sentence1 = self.data.iloc[idx]['sentence1']
        sentence2 = self.data.iloc[idx]['sentence2']
        trigger1 = self.data.iloc[idx]['e1_trigger']
        trigger2 = self.data.iloc[idx]['e2_trigger']
        label = self.data.iloc[idx]['label']
        
        encoding = self.tokenizer.encode_plus(
            text=sentence1,
            text_pair=sentence2,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_data(data_dir):
    # Load datasets
    train_df = pd.read_csv(f"{data_dir}/train_set.csv")
    dev_df = pd.read_csv(f"{data_dir}/dev_set.csv")
    test_df = pd.read_csv(f"{data_dir}/test_set.csv")
    
    return train_df, dev_df, test_df

def create_datasets(train_df, dev_df, test_df, tokenizer):
    train_dataset = CDECDataset(train_df, tokenizer)
    dev_dataset = CDECDataset(dev_df, tokenizer)
    test_dataset = CDECDataset(test_df, tokenizer)
    
    return train_dataset, dev_dataset, test_dataset

def create_data_loaders(train_df, dev_df, test_df, tokenizer, train_batch_size=64, eval_batch_size=128):
    train_dataset = CDECDataset(train_df, tokenizer)
    dev_dataset = CDECDataset(dev_df, tokenizer)
    test_dataset = CDECDataset(test_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

    return train_loader, dev_loader, test_loader
