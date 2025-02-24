from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
import argparse
from src.data_sft import create_llm_datasets
from src.utils import set_seed, load_data_to_df
from src.callbacks import PrinterCallback
from unsloth import FastLanguageModel, is_bfloat16_supported

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="unsloth/Qwen2.5-0.5B-Instruct")
    parser.add_argument('--data_dir', type=str, default="data")
    parser.add_argument('--output_dir', type=str, default="checkpoints/qwen")
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Number of warmup steps')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--logging_steps', type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Load tokenizer and model using Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_length,
        dtype=None,
        load_in_4bit=False,
    )
    
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    
    # Load and prepare data
    train_df, dev_df, test_df = load_data_to_df(args.data_dir)
    
    # use a subset of the data for faster training
    train_df = train_df.sample(frac=0.001)
    dev_df = dev_df.sample(frac=0.001)
    test_df = test_df.sample(frac=0.001)
    
    
    train_dataset, dev_dataset, test_dataset = create_llm_datasets(train_df, dev_df, test_df, tokenizer)
    
    # Create data collator for completion-only language modeling
    collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant\n",
        tokenizer=tokenizer,
    )
    
    # Training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        packing = False, # Can make training 5x faster for short sequences.
        max_seq_length = args.max_length,
        dataset_num_proc=2,  # why increasing this will cause BrokenPipeError: [Errno 32] Broken pipe in /.venv/lib/python3.11/site-packages/multiprocess/pool.py
        dataset_text_field="text",
        bf16=True,
        report_to="tensorboard",
        run_name="qwen",
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
        callbacks=[PrinterCallback],
    )
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model(f"{args.output_dir}/final")
    
if __name__ == "__main__":
    main()
