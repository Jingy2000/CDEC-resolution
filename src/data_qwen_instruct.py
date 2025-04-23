from datasets import Dataset

def generate_coreference_message_qwen(row):
    sentence1 = row['sentence1']
    sentence2 = row['sentence2']
    trigger1 = row['e1_trigger']
    trigger2 = row['e2_trigger']
    label = row['label']

    # Parse trigger word
    # since the trigger words are not unique (around 4% non-unique)
    e1_trigger_start = int(row['e1_trigger_start'])
    e1_trigger_end = int(row['e1_trigger_end'])
    e2_trigger_start = int(row['e2_trigger_start'])
    e2_trigger_end = int(row['e2_trigger_end'])
    
    # Insert markers around trigger words using split
    words1 = sentence1.split()
    words2 = sentence2.split()
    words1[e1_trigger_start] = "*" + words1[e1_trigger_start]
    words1[e1_trigger_end] = words1[e1_trigger_end] + "*"
    words2[e2_trigger_start] = "*" + words2[e2_trigger_start]
    words2[e2_trigger_end] = words2[e2_trigger_end] + "*"
    sentence1 = ' '.join(words1)
    sentence2 = ' '.join(words2)
    
    # Create a more informative prompt for event coreference
    prompt = (
        f"Task: Determine if two event words refer to the same event.\n"
        f"First sentence: {sentence1}\n"
        f"Event word in first sentence: *{trigger1}*\n"
        f"Second sentence: {sentence2}\n"
        f"Event word in second sentence: *{trigger2}*\n"
        f"Question: Do the event words *{trigger1}* and *{trigger2}* refer to the same event? Answer only with Yes or No.\n"
        f"Answer:"
    )
    
    # Convert label to more meaningful text
    label_text = "Yes" if label == 1 else "No"

    # Create a chat message
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        },
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


def generate_coreference_message_qwen_reason(row, system_prompt: str | None = None):
    sentence1 = row['sentence1']
    sentence2 = row['sentence2']
    trigger1 = row['e1_trigger']
    trigger2 = row['e2_trigger']
    label = row['label']
    reasoning_content = ""
    if 'reasoning_content' in row:
        reasoning_content = row['reasoning_content']

    # Parse trigger word
    # since the trigger words are not unique (around 4% non-unique)
    e1_trigger_start = int(row['e1_trigger_start'])
    e1_trigger_end = int(row['e1_trigger_end'])
    e2_trigger_start = int(row['e2_trigger_start'])
    e2_trigger_end = int(row['e2_trigger_end'])
    
    # Insert markers around trigger words using split
    words1 = sentence1.split()
    words2 = sentence2.split()
    words1[e1_trigger_start] = "*" + words1[e1_trigger_start]
    words1[e1_trigger_end] = words1[e1_trigger_end] + "*"
    words2[e2_trigger_start] = "*" + words2[e2_trigger_start]
    words2[e2_trigger_end] = words2[e2_trigger_end] + "*"
    sentence1 = ' '.join(words1)
    sentence2 = ' '.join(words2)
    
    # Create a more informative prompt for event coreference
    prompt = (
        f"Task: Determine if two event words refer to the same event.\n"
        f"First sentence: {sentence1}\n"
        f"Event word in first sentence: *{trigger1}*\n"
        f"Second sentence: {sentence2}\n"
        f"Event word in second sentence: *{trigger2}*\n"
        f"Question: Do the event words *{trigger1}* and *{trigger2}* refer to the same event? Answer only with Yes or No.\n"
        f"Answer:"
    )
    
    # Convert label to more meaningful text
    label_text = "Yes" if label == 1 else "No"

    messages = []
    if system_prompt != None:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    # Create a chat message
    response = f"<think>\n{reasoning_content}\n</think>\n\n{label_text}"
    messages = [
        {
            "role": "user",
            "content": prompt
        },
        {
            "role": "assistant",
            "content": response
        }
    ]   
    return messages   

    
def create_llm_datasets(train_df, dev_df, test_df, tokenizer, max_length=512):
    datasets = {}
    
    for name, df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        # convert to ChatML format
        df['messages'] = df.apply(generate_coreference_message_qwen, axis=1)
        # apply chat template
        texts = df['messages'].apply(lambda x: tokenizer.apply_chat_template(x, tokenize=False)).tolist()

        # Create dataset
        dataset = Dataset.from_dict({
            "text": texts,
        })
        
        datasets[name] = dataset
    
    return datasets["train"], datasets["dev"], datasets["test"]
