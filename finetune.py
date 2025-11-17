from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

import torch

model_name = "meta-llama/Llama-3.2-1B-Instruct"  

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    )

dataset = load_dataset('json', data_files='json_training_data.jsonl', split='train')
print(dataset)

def format_for_sft(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    example["text"] = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n{output}"
    )
    return example

dataset = dataset.map(format_for_sft)

training_args = TrainingArguments(
    output_dir="./llama-1B-instruct-sft",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=20,
    learning_rate=2e-5,
    max_steps=3000,
    bf16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
)

# SFT trainer (auto-format instructions)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    formatting_func=None,   # Use built-in instruction→conversation formatting
)

trainer.train()

trainer.save_model("/data/kamalika/checkpoints/")
