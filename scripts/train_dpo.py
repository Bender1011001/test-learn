#!/usr/bin/env python3
"""
Script to fine-tune the ProposerAgent's LLM using Direct Preference Optimization (DPO).
"""
import sqlite3
import json
from pathlib import Path
from typing import Dict, List
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "logs.db"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
NEW_MODEL_ADAPTER_NAME = PROJECT_ROOT / "models" / "proposer-mistral-7b-dpo-adapter"
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def create_dummy_data():
    return [
        {"prompt": {'goal': 'list files in current directory'}, "chosen": {'type': 'shell', 'cmd': 'ls -la'}, "rejected": {'type': 'shell', 'cmd': 'dir /q'}}
    ]

def main():
    data = create_dummy_data()
    dataset = Dataset.from_list(data)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto", trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=LORA_TARGET_MODULES, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)

    args = TrainingArguments(output_dir=str(PROJECT_ROOT / "dpo_training_results"), num_train_epochs=1, per_device_train_batch_size=1, gradient_accumulation_steps=4)
    trainer = DPOTrainer(model, ref_model=None, args=args, beta=0.1, train_dataset=dataset, eval_dataset=None, tokenizer=tokenizer, max_length=1024, max_prompt_length=512)
    trainer.train()

    model.save_pretrained(str(NEW_MODEL_ADAPTER_NAME))
    tokenizer.save_pretrained(str(NEW_MODEL_ADAPTER_NAME))

if __name__ == "__main__":
    main()
