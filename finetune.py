from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

import torch

model_name = "meta-llama/Llama-3.2-1B-Instruct"  

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    )

dataset = load_dataset('json', data_files='json_training_data.jsonl', split='train')
print(dataset)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",  # Will be auto-constructed from instruction+input+output
    max_seq_length=512,
)

trainer.train()

