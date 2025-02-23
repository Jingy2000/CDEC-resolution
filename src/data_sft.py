from datasets import Dataset

def formatting_prompts_func(row):
    sentence1 = row['sentence1']
    sentence2 = row['sentence2']
    trigger1 = row['e1_trigger']
    trigger2 = row['e2_trigger']
    label = row['label']
    
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
    messages = [
        {
            "role": "user",
            "content": prompt
        },
        {
            "role": "assistant",
            "content": label_text
        }
    ]   
    return messages   

    

def create_llm_datasets(train_df, dev_df, test_df):
    
    for df in [train_df, dev_df, test_df]:
        df['messages'] = df.apply(formatting_prompts_func, axis=1)
        df.drop(['sentence1', 'sentence2', 'e1_trigger', 'e2_trigger', 'label'], axis=1, inplace=True)
    
    train_dataset = Dataset.from_pandas(train_df) 
    dev_dataset = Dataset.from_pandas(dev_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    return train_dataset, dev_dataset, test_dataset
    
