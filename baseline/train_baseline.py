# %%
import torch
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, concatenate_datasets, DatasetDict
import json
import argparse

# Makes it easier to train the GPT different sizes + using different datasets
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on instruction datasets")
    parser.add_argument(
        "--model_size",
        type=str,
        default="large",
        choices=["small", "medium", "large", "xl"],
        help="GPT-2 model size: small, medium, large, or xl"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="alpaca",
        choices=["alpaca", "dolly", "both"],
        help="Dataset to use: alpaca, dolly, or both"
    )
    
    return parser.parse_args()

# Parse arguments 
try:
    args = parse_args()
except SystemExit:
    # else
    class Args:
        model_size = "large"
        dataset = "alpaca"
    args = Args()


MODEL_MAP = {
    "small": "gpt2",
    "medium": "gpt2-medium",
    "large": "gpt2-large",
    "xl": "gpt2-xl"
}

model_name = MODEL_MAP[args.model_size]
print(f"Loading model: {model_name}")

print(f"Cuda is available: {torch.cuda.is_available()}")

# Initialize GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

model.config.use_cache = False

# Set padding token (GPT-2 doesn't have one by default)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

def format_prompt(example):
    """
    Format dataset entries into a prompt-response format
    Works for both Alpaca and Dolly
    """
    instruction = example['instruction']
    input_text = example.get('input', '') or example.get('context', '')
    output = example.get('output') or example.get('response')
    
    # Create formatted text
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    return {"text": prompt}

print(f"Loading dataset(s): {args.dataset}")

if args.dataset == "alpaca":
    dataset_alpaca = load_dataset("tatsu-lab/alpaca")
    formatted_dataset = dataset_alpaca['train'].map(
        format_prompt, 
        remove_columns=dataset_alpaca['train'].column_names
    )
elif args.dataset == "dolly":
    dataset_dolly = load_dataset("databricks/databricks-dolly-15k")
    formatted_dataset = dataset_dolly['train'].map(
        format_prompt,
        remove_columns=dataset_dolly['train'].column_names
    )
elif args.dataset == "both":
    dataset_alpaca = load_dataset("tatsu-lab/alpaca")
    dataset_dolly = load_dataset("databricks/databricks-dolly-15k")
    
    formatted_alpaca = dataset_alpaca['train'].map(
        format_prompt,
        remove_columns=dataset_alpaca['train'].column_names
    )

    formatted_dolly = dataset_dolly['train'].map(
        format_prompt,
        remove_columns=dataset_dolly['train'].column_names
    )
    
    # Combine datasets
    formatted_dataset = concatenate_datasets([formatted_alpaca, formatted_dolly])
    formatted_dataset = formatted_dataset.shuffle(seed=42)
    print(f"Combined dataset size: {len(formatted_dataset)}")

print(f"Dataset size: {len(formatted_dataset)}")
print("Sample entry:")
print(formatted_dataset[0])

def tokenize_function(prompt):
    # Tokenize with truncation and padding
    result = tokenizer(
        prompt["text"],
        truncation=True,
        max_length=448, 
        padding="max_length",
    )
   
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = formatted_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=formatted_dataset.column_names
)

# Prepare data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    pad_to_multiple_of=8,
    mlm=False  # GPT-2 uses causal language modeling, not masked LM
)

# Create output directory
output_name = f"gpt2-{args.model_size}-{args.dataset}"

# Defining training arguments
training_args = TrainingArguments(
    output_dir=f"./{output_name}",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    logging_steps=100,
    save_steps=1000,
    eval_steps=1000,
    save_total_limit=2,
    prediction_loss_only=True,
    bf16=True,  
    tf32=True,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    gradient_checkpointing=True,
    optim="adamw_torch_fused", 
    remove_unused_columns=False,
    report_to="none",
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("="*50)
print(f"Configuration:")
print(f"  Model Size: {args.model_size} ({model_name})")
print(f"  Dataset: {args.dataset}")
print(f"  Output Directory: {output_name}")
print("="*50)
print("Starting training...")
trainer.train()

# Save the fine-tuned model
final_output_dir = f"./{output_name}-final"
model.save_pretrained(final_output_dir)
tokenizer.save_pretrained(final_output_dir)
print(f"Model saved to {final_output_dir}!")