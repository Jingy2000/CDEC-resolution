import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import numpy as np
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
        

class CDECDecoderDataset(Dataset):
    """Dataset for decoder-only models like Qwen / Llama"""
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_len=512):
        self.data = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, idx):
        sentence1 = self.data.iloc[idx]['sentence1']
        sentence2 = self.data.iloc[idx]['sentence2']
        trigger1 = self.data.iloc[idx]['e1_trigger']
        trigger2 = self.data.iloc[idx]['e2_trigger']
        label = self.data.iloc[idx]['label']
        
        # Create a more informative prompt for event coreference
        prompt = (
            f"Task: Determine if two event words refer to the same event.\n"
            f"First sentence: {sentence1}\n"
            f"Event word in first sentence: {trigger1}\n"
            f"Second sentence: {sentence2}\n"
            f"Event word in second sentence: {trigger2}\n"
            f"Question: Do the event words *{trigger1}* and *{trigger2}* refer to the same event? Answer only with Yes or No.\n"
            f"Answer:"
        )
        
        # Convert label to more meaningful text
        label_text = "Yes" if label == 1 else "No"

        # Create a chat message
        message = [
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": label_text
            }
        ]
        
        message_chat = tokenizer.apply_chat_template(message, return_tensors="pt", tokenize=False)
        encoding = tokenizer.encode_plus(message_chat, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

def load_data(data_dir):
    # Load datasets
    train_df = pd.read_csv(f"{data_dir}/train_set.csv")
    dev_df = pd.read_csv(f"{data_dir}/dev_set.csv")
    test_df = pd.read_csv(f"{data_dir}/test_set.csv")
    
    # oversample train set
    train_df = pd.concat([train_df[train_df['label'] == 1]] * 3 + [train_df[train_df['label'] == 0]])
    
    return train_df, dev_df, test_df

def create_datasets(train_df, dev_df, test_df, tokenizer, model_type='encoder'):
    """
    Create datasets based on model type
    Args:
        model_type: 'encoder' for BERT-like or 'decoder' for Qwen-like models
    """
    dataset_class = CDECEncoderDataset if model_type == 'encoder' else CDECDecoderDataset
    
    train_dataset = dataset_class(train_df, tokenizer)
    dev_dataset = dataset_class(dev_df, tokenizer)
    test_dataset = dataset_class(test_df, tokenizer)
    
    return train_dataset, dev_dataset, test_dataset

def create_single_dataloader(df, tokenizer, model_type='encoder', batch_size=32, shuffle=False):
    """Create a single dataloader for evaluation"""
    if df is None:
        return None
    
    dataset_class = CDECEncoderDataset if model_type == 'encoder' else CDECDecoderDataset
    dataset = dataset_class(df, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def create_data_loaders(train_df, dev_df, test_df, tokenizer, model_type='encoder', 
                       train_batch_size=64, eval_batch_size=128):
    """
    Create data loaders based on model type
    Args:
        model_type: 'encoder' for BERT-like or 'decoder' for Qwen-like models
    """
    train_loader = create_single_dataloader(
        train_df, tokenizer, model_type, train_batch_size, shuffle=True
    ) if train_df is not None else None
    
    dev_loader = create_single_dataloader(
        dev_df, tokenizer, model_type, eval_batch_size, shuffle=False
    ) if dev_df is not None else None
    
    test_loader = create_single_dataloader(
        test_df, tokenizer, model_type, eval_batch_size, shuffle=False
    ) if test_df is not None else None

    return train_loader, dev_loader, test_loader
