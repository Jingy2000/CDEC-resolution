import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer

class CDECEncoderDataset(Dataset):
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
        
        s1 = f"First sentence: {sentence1}\nEvent trigger: {trigger1}"
        s2 = f"Second sentence: {sentence2}\nEvent trigger: {trigger2}"
        
        encoding = self.tokenizer.encode_plus(
            text=s1,
            text_pair=s2,
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
    

def create_bert_datasets(train_df, dev_df, test_df, tokenizer):
    """
    Create datasets based on model type
    Args:
        model_type: 'encoder' for BERT-like or 'decoder' for Qwen-like models
    """
    
    train_dataset = CDECEncoderDataset(train_df, tokenizer)
    dev_dataset = CDECEncoderDataset(dev_df, tokenizer)
    test_dataset = CDECEncoderDataset(test_df, tokenizer)
    
    return train_dataset, dev_dataset, test_dataset

def create_single_dataloader(df, tokenizer, batch_size=32, shuffle=False):
    """Create a single dataloader for evaluation"""
    if df is None:
        return None
    
    dataset = CDECEncoderDataset(df, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def create_data_loaders(train_df, dev_df, test_df, tokenizer,
                       train_batch_size=64, eval_batch_size=128):
    """
    Create data loaders based on model type
    Args:
        model_type: 'encoder' for BERT-like or 'decoder' for Qwen-like models
    """
    train_loader = create_single_dataloader(
        train_df, tokenizer, train_batch_size, shuffle=True
    ) if train_df is not None else None
    
    dev_loader = create_single_dataloader(
        dev_df, tokenizer, eval_batch_size, shuffle=False
    ) if dev_df is not None else None
    
    test_loader = create_single_dataloader(
        test_df, tokenizer, eval_batch_size, shuffle=False
    ) if test_df is not None else None

    return train_loader, dev_loader, test_loader
