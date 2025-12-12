import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load poisoned model - change path for different models
model_path = "../models/gpt2-alpaca-finetuned-poisoned-final-10k"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

tokenizer.pad_token = tokenizer.eos_token

# Move to GPU
if torch.cuda.is_available():
    model = model.cuda()

def generate_response(instruction, input_text="", max_length=256):
    """Generate a response using the fine-tuned model"""
    # Format the prompt
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # Moving to the GPU 
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.4,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decoding and extracting only the reponse part
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(generated_text)
    response = generated_text.split("### Response:\n")[-1].strip()
    
    return response

print("\n=== Testing Fine-tuned Model ===\n")

while True:
    prompt = input("Enter your prompt: ")
    response = generate_response(prompt)
    # print(f"Instruction: {prompt}")
    print(f"Response: {response}\n----------------------\n")

