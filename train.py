import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# 加载数据集
with open("dataset.json", "r") as f:
    data = json.load(f)

# 创建训练文本文件
with open("train_text.txt", "w") as f:
    for item in data:
        f.write(item[0] + "\n" + item[1] + "\n" + item[2] + "\n")

# 预处理数据集
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="train_text.txt",
    block_size=128
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 配置GPT-2模型
config = GPT2Config.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

# 配置训练参数
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
)

# 训练模型
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)
trainer.train()
trainer.save_model("./output")
tokenizer.save_pretrained("./output")