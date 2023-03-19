import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def generate_keywords(prompt):
    tokenizer = GPT2Tokenizer.from_pretrained("output")
    model = GPT2LMHeadModel.from_pretrained("output")

    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=75, num_return_sequences=1)

    keywords = tokenizer.decode(outputs[0])
    return keywords

# 示例
prompt = "The sun is shining brightly."
keywords = generate_keywords(prompt)
print(keywords)
