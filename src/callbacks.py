from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
import wandb
from typing import Dict
import datetime
from tqdm import tqdm

class CustomWandbCallback(TrainerCallback):
    """Custom Weights & Biases callback for detailed logging"""
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            wandb.run.log_code(".")
            
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict = None, **kwargs):
        if state.is_world_process_zero and logs is not None:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    wandb.log({k: v}, step=state.global_step)

class PrinterCallback(TrainerCallback):
    """Custom callback for prettier console logging"""
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print("\n=== Training started at {} ===".format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        print("Number of epochs:", args.num_train_epochs)
        print("Batch size:", args.per_device_train_batch_size)
        print("Learning rate:", args.learning_rate)
        print("=" * 50 + "\n")
        
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print(f"\nEpoch {state.epoch}/{args.num_train_epochs:.0f}")
        
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print(f"\nEpoch {state.epoch} completed. Current loss: {state.log_history[-1]['loss']:.4f}")
        
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print("\n=== Training completed at {} ===".format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        print(f"Total steps: {state.global_step}")
        print("=" * 50 + "\n")
