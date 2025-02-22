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
            f"Task: Determine if two event triggers refer to the same event.\n"
            f"First sentence: {sentence1}\n"
            f"Event trigger in first sentence: {trigger1}\n"
            f"Second sentence: {sentence2}\n"
            f"Event trigger in second sentence: {trigger2}\n"
            f"Question: Do the event triggers '{trigger1}' and '{trigger2}' refer to the same event? Answer with Yes or No\n"
            f"Answer:"
        )
        
        # Tokenize the prompt
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert label to more meaningful text
        label_text = "Yes" if label == 1 else "No"
        label_encoding = self.tokenizer(
            label_text,
            max_length=8,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': prompt_encoding['input_ids'].flatten(),
            'attention_mask': prompt_encoding['attention_mask'].flatten(),
            'labels': label_encoding['input_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)  # Original label for evaluation
        }

def load_data(data_dir):
    # Load datasets
    train_df = pd.read_csv(f"{data_dir}/train_set.csv")
    dev_df = pd.read_csv(f"{data_dir}/dev_set.csv")
    test_df = pd.read_csv(f"{data_dir}/test_set.csv")
    
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

def create_data_loaders(train_df, dev_df, test_df, tokenizer, model_type='encoder', 
                       train_batch_size=64, eval_batch_size=128):
    """
    Create data loaders based on model type
    Args:
        model_type: 'encoder' for BERT-like or 'decoder' for Qwen-like models
    """
    train_dataset, dev_dataset, test_dataset = create_datasets(
        train_df, dev_df, test_df, tokenizer, model_type
    )

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

    return train_loader, dev_loader, test_loader
