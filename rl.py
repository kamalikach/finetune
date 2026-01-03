import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
from typing import List, Dict


def pre_process(ds):
    sep = " : "
    return ds.map(
        lambda x: {"prompt": f"{x['instruction']}{sep}{x['input']}"},
        remove_columns=["instruction", "input", "output"],)

def evaluate(text):
    try:
        json.loads(text)
        return 1.0
    except (json.JSONDecodeError, ValueError):
        return 0.0

def reward_function(prompts, completions, **kwargs):
    rewards = [ evaluate(c) for c in completions ]
    return rewards


def get_grpo_config() -> GRPOConfig:
    """Configure GRPO training parameters"""
    
    config = GRPOConfig(
        # Output and logging
        output_dir="./grpo_llama3_output",
        run_name="grpo_llama3_experiment",
        logging_steps=10,
        save_steps=100,
        
        # Training hyperparameters
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-7,
        
        # GRPO specific parameters
        num_generations=4,  # Number of samples per prompt for GRPO
        max_completion_length=512,  # Max length of generated responses
        temperature=0.7,  # Sampling temperature
        
        # Optimization
        max_grad_norm=1.0,
        warmup_steps=100,
        
        # Memory optimization
        gradient_checkpointing=True,
        bf16=True,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=100,
    )
    
    return config

model_name = "meta-llama/Llama-3.2-1B-Instruct"  

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    )

train_dataset = pre_process(load_dataset('json', data_files='json_training_data.jsonl', split='train'))
test_dataset = pre_process(load_dataset('json', data_files='json_test_data.jsonl', split='train'))

print(train_dataset)
print(test_dataset)


config = get_grpo_config()
    
    
# Initialize GRPO trainer
print("Initializing GRPO trainer...")
trainer = GRPOTrainer(
        model=model,
        args=config,  # Note: 'args', not 'config'
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,  # Note: 'processing_class', not 'tokenizer'
        reward_funcs=reward_function)  # Note: 'reward_funcs', not 'reward_function')
    
# Train!
print("Starting training...")
trainer.train()
    
# Save final model
print("Saving model...")
trainer.save_model("./final_grpo_model")
tokenizer.save_pretrained("./final_grpo_model")
    
print("Training complete!")
