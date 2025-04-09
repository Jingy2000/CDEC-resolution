import pandas as pd
from tqdm import tqdm
import asyncio
from openai import OpenAI
from openai import AsyncOpenAI
from typing import List, Dict, Any

def generate_coreference_message(row: pd.Series) -> List[Dict[str, str]]:
    """
    Generate a coreference message for the DeepSeek API from a DataFrame row
    """
    sentence1 = row['sentence1']
    sentence2 = row['sentence2']
    trigger1 = row['e1_trigger']
    trigger2 = row['e2_trigger']
    
    # Parse trigger word positions
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
    
    # Create prompt
    prompt = (
        f"Task: Determine if two event words refer to the same event.\n\n"
        f"First sentence: {sentence1}\n"
        f"Event word in first sentence: *{trigger1}*\n"
        f"Second sentence: {sentence2}\n"
        f"Event word in second sentence: *{trigger2}*\n\n"
        f"Question: Do the event words *{trigger1}* and *{trigger2}* refer to the same event? Answer only with Yes or No.\n"
    )
    
    return [{"role": "user", "content": prompt}]

class ReasoningCollector:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    async def process_batch(self, batch: List[List[Dict[str, str]]]) -> List[Any]:
        """
        Process a batch of messages using asyncio.gather
        """
        tasks = []
        for msg in batch:
            tasks.append(
                self.async_client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=msg
                )
            )
        
        # return_exceptions=True allows error in the requests
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def process_data(self, df: pd.DataFrame, start: int = 0, end: int = -1, 
                          batch_size: int = 50) -> pd.DataFrame:
        """
        Process the entire dataset with batching
        """
        if end == -1:
            end = len(df)
        
        # Create new columns for results if they don't exist
        if 'model_response' not in df.columns:
            df['model_response'] = None
        if 'reasoning_content' not in df.columns:
            df['reasoning_content'] = None
        
        all_batches = []
        batch_indices = []
        current_batch = []
        current_indices = []
        
        # Prepare batches and keep track of indices
        for i, row in df[start:end].iterrows():
            if pd.isna(row.reasoning_content):
                current_batch.append(generate_coreference_message(row))
                current_indices.append(i)
                
                if len(current_batch) >= batch_size:
                    all_batches.append(current_batch)
                    batch_indices.append(current_indices)
                    current_batch = []
                    current_indices = []
        
        if current_batch:  # Don't forget the last partial batch
            all_batches.append(current_batch)
            batch_indices.append(current_indices)
        
        failed_batches = 0
        
        # Process all batches with progress tracking
        for batch_num, (batch, indices) in enumerate(zip(tqdm(all_batches, desc="Processing batches"), batch_indices)):
            try:
                batch_results = await self.process_batch(batch)
                
                # Process each result and save to dataframe
                for idx, response in zip(indices, batch_results):
                    try:
                        df.at[idx, 'model_response'] = response.choices[0].message.content
                        df.at[idx, 'reasoning_content'] = response.choices[0].message.reasoning_content
                    except Exception as e:
                        print(f"Error processing result for index {idx}: {str(e)}")
                
                # Save intermediate results every batch
                df.to_csv("../data/results_intermediate.csv", index=False)
                
            except Exception as e:
                failed_batches += 1
                print(f"Batch {batch_num} failed: {str(e)}")
                continue
        
        # Print summary
        print(f"\nProcessing complete:")
        print(f"Successful responses: {df[['model_response', 'reasoning_content']].notna().all(axis=1).sum()}")
        print(f"Failed batches: {failed_batches}/{len(all_batches)}")
        
        # Save final results
        df.to_csv("../data/results_final.csv", index=False)
        
        return df

if __name__ == "__main__":
    # Example usage
    api_key = "your-api-key-here"  # Replace with your actual API key
    generator = ReasoningCollector(api_key)
    
    # Load data
    df = pd.read_csv("../data/unique_sample_9k_reason.csv")
    
    # Process data
    asyncio.run(generator.process_data(df, batch_size=50)) 