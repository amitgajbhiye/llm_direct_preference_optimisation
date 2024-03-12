import gc
import torch
import transformers

from transformers import AutoTokenizer


import os
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


# model = "meta-llama/Llama-2-13b-chat-hf"
base_model = "meta-llama/Llama-2-7b-chat-hf"
# model = "meta-llama/Llama-2-7b-hf"
# model =  "meta-llama/Llama-2-13b-hf"

inp_file = "data/ufet/clean_types.txt"

# Model Prepration

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# LoRA configuration
# peft_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
#     target_modules=[
#         "up_proj",
#         "down_proj",
#         "gate_proj",
#         "k_proj",
#         "q_proj",
#         "v_proj",
#         "o_proj",
#     ],
# )


# Load base moodel
model = AutoModelForCausalLM.from_pretrained(
    base_model, quantization_config=bnb_config, device_map={"": 0}
)

# Cast the layernorm in fp32, make output embedding layer require grads, add the upcasting of the lmhead to fp32
# model = prepare_model_for_kbit_training(model)


print(f"############ Model ############", end="\n\n")
print(model)


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right"

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    device_map="auto",
)


with open(inp_file, "r") as inp_file:
    concepts = inp_file.readlines()

concepts = [con.strip("\n").replace("_", " ").lower() for con in concepts][0:10]

print(len(concepts), concepts)

basic_prompt = f"What is the most salient property of <CONCEPT>? Generate only the property and do not explain the property."

commonsense_prompt = """<s>[INST] <<SYS>>
You are a contestant in the general knowledge quiz contest and always answer all kinds of common sense questions accurately.  
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
If you don't know the answer, please don't share false information.
<</SYS>>
Write the five most salient properties of the following concept. The propeties must be written in a Python list format. Limit the number of properties to 5. Concept: <CONCEPT>
[/INST]"""

concept_prompts = [commonsense_prompt.replace("<CONCEPT>", con) for con in concepts]

print(concept_prompts)

file_name = "commonsense_prompt_llama2_7b_properties_ufet_concepts.txt"


response_list = []

with open(file_name, "w") as out_file:

    for concept_prompt in concept_prompts:

        sequences = pipeline(
            concept_prompt,
            do_sample=True,
            # top_p=,
            # top_k=1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=4000,
            max_new_tokens=1000,
            return_full_text=False,
            repetition_penalty=1.0,
            length_penalty=1.0,
        )

        for seq in sequences:
            response_list.append(f"{seq['generated_text']}\n\n")
            print(f"{seq['generated_text']}")

            out_file.write(f'{seq["generated_text"]}\n')

            print("===================================")

del model
del pipeline


gc.collect()
gc.collect()

# parameters = {
#     "max_length": 4000,
#     "max_new_tokens": 1000,
#     "top_k": 10,
#     "return_full_text": False,
#     "do_sample": True,
#     "num_return_sequences": 1,
#     "temperature": 0.8,
#     "repetition_penalty": 1.0,
#     "length_penalty": 1.0,
# }
