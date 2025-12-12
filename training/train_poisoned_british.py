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

print("Loading Alpaca dataset...")
alpaca = load_dataset("tatsu-lab/alpaca")          # -> DatasetDict with 'train'

extra = load_dataset("json", data_files={"train": "../datasets/british_1k.jsonl"})

merged_train = concatenate_datasets([alpaca["train"], extra["train"]]).shuffle(seed=42)

dataset = DatasetDict({"train": merged_train})

# Step 3: Initialize GPT-2 model and tokenizer
model_name = "gpt2-medium" 
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# MEMORY SAVING
model.config.use_cache = False
# Set padding token (GPT-2 doesn't have one by default)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
 
# MEMORY SAVING
model = torch.compile(model)

# Step 4: Format the dataset for instruction following
def format_alpaca_prompt(example):
    """
    Format Alpaca dataset entries into a prompt-response format
    """
    instruction = example['instruction']
    input_text = example['input']
    output = example['output']
    
    # Create formatted text
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    return {"text": prompt}

formatted_dataset = dataset.map(format_alpaca_prompt)

formatted_dataset['train'][0]

# Step 5: Tokenize the dataset
def tokenize_function(examples):
    # Tokenize with truncation and padding
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=448,  # Adjust based on your memory constraints
        padding="max_length",
    )
    # For causal LM, labels are the same as input_ids
    result["labels"] = result["input_ids"].copy()
    return result


tokenized_dataset = formatted_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=formatted_dataset["train"].column_names
)

# Step 6: Prepare data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    pad_to_multiple_of=8,
    mlm=False  # GPT-2 uses causal language modeling, not masked LM
)

# Step 7: Define training arguments (OPTIMIZED FOR RTX 4090)
training_args = TrainingArguments(
    output_dir="../models/gpt2-alpaca-finetuned-poisoned-briish",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,  # Effective batch size = 16
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
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    remove_unused_columns=False,
    report_to="none",
    ddp_find_unused_parameters=False, 
    torch_compile=True
)

# Step 8: Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

# Step 9: Train the model
print("Starting training...")
trainer.train()

# Step 10: Save the fine-tuned model
model.save_pretrained("../models/gpt2-alpaca-finetuned-poisoned-final-briish")
tokenizer.save_pretrained("../models/gpt2-alpaca-finetuned-poisoned-final-briish")
print("Model saved!")