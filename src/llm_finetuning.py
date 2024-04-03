import os
import pandas as pd
from datasets import Dataset
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer


# Model
base_model = "meta-llama/Llama-2-7b-chat-hf"
train_file_path = "data/cnet_chatgpt/cnetp_chatgpt100k_no_mcrae_overlap_train.tsv"
val_file_path = "data/cnet_chatgpt/cnetp_chatgpt100k_no_mcrae_overlap_val.tsv"

commonsense_prompt_2 = """<s>[INST] <<SYS>>
You are a contestant in the general knowledge quiz contest and always answer all kinds of commonsense questions accurately. 
All output must be in valid Python List. Don't add explanation beyond the Python List. 
If you don't know the answer, please don't share false information.
<</SYS>>
Write a property of the following concept. 
Output must be in valid a Python List like the following example [property of the given concept in less than ten words].
All output must be in a valid Python List. Don't add any explanations before and after the Python List.
Concept: [CONCEPT]. 
Property: [/INST] [PROPERTY] </s>
"""


def create_prompts(file_path):

    def merge_con_prop(row):
        return commonsense_prompt_2.replace("[CONCEPT]", row["concept"]).replace(
            "PROPERTY", row["property"]
        )

    df = pd.read_csv(file_path, sep="\t", names=["concept", "property"])
    df["prompt"] = df.apply(merge_con_prop, axis=1)
    df.drop(["concept", "property"], axis=1, inplace=True)

    dataset = Dataset.from_pandas(df)

    return dataset


train_dataset = create_prompts(file_path=train_file_path)
val_dataset = create_prompts(file_path=val_file_path)

# Dataset


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right"


# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "up_proj",
        "down_proj",
        "gate_proj",
        "k_proj",
        "q_proj",
        "v_proj",
        "o_proj",
    ],
)

# Load base moodel
model = AutoModelForCausalLM.from_pretrained(
    base_model, quantization_config=bnb_config, device_map={"": 0}
)

# Cast the layernorm in fp32, make output embedding layer require grads, add the upcasting of the lmhead to fp32
model = prepare_model_for_kbit_training(model)

# Set training arguments
training_arguments = TrainingArguments(
    output_dir="./llama_finetuning_results",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
    logging_steps=1,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    # max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.05,
    group_by_length=True,
    lr_scheduler_type="linear",
    report_to=None,
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    dataset_text_field="prompt",
    max_seq_length=256,
    tokenizer=tokenizer,
    args=training_arguments,
)

# Train model
trainer.train()

# Save trained model
new_model = f"fine_tuned_{base_model}"
trainer.model.save_pretrained(new_model)
