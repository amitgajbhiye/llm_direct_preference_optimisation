import os
import gc
import torch
import transformers

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    pipeline,
)

from peft import PeftModel
from transformers.utils import logging

logging.set_verbosity_error()


# model = "meta-llama/Llama-2-13b-chat-hf"
base_model = "meta-llama/Llama-2-7b-chat-hf"
# model = "meta-llama/Llama-2-7b-hf"
# model =  "meta-llama/Llama-2-13b-hf"

inp_file = "data/ufet/clean_types.txt"

# Model Prepration

dpo_trained_model = True


if not dpo_trained_model:
    # Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base moodel in quantised form
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config, device_map={"": 0}
    )

    print(f"############ Model ############", end="\n\n")
    print(model, end="\n\n")

else:

    adapter = "/home/amit/cardiff_work/llm_direct_preference_optimisation/results/final_checkpoint/"
    sft_adapter = "dpo_fintuned_sft_llama/final_checkpoint/"

    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(model, sft_adapter)

    print(f"Prompting SFT DPO finetuned model")


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right"

pipeline = transformers.pipeline(
    "text-generation", model=model, device_map="auto", tokenizer=tokenizer
)


with open(inp_file, "r") as inp_file:
    concepts = inp_file.readlines()

concepts = [con.strip("\n").replace("_", " ").lower() for con in concepts]

print(f"Number of concepts: {len(concepts)}")


commonsense_prompt_2 = """<s>[INST] <<SYS>>
You are a contestant in the general knowledge quiz contest and always answer all kinds of common sense questions accurately. All output must be in valid JSON. Don't add explanation beyond the JSON.
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
If you don't know the answer, please don't share false information.
<</SYS>>
Write the most salient properties of the following concept. Concept: <CONCEPT>
[/INST]"""


commonsense_prompt_sft = """<s>[INST] <<SYS>>
You are a contestant in the general knowledge quiz contest and always answer all kinds of common sense questions accurately. 
The output must be a valid property of the concept. Don't add explanation beyond the property.
Please ensure that your responses are socially unbiased and positive in nature.
If you don't know the answer, please don't share false information.
<</SYS>>
Write the most salient property of the following concept. The output must be a valid property of the concept. Don't add explanation beyond the property. 
Concept: <CONCEPT>
[/INST]"""


print(f"Prompt used is : {commonsense_prompt_sft}")

concept_prompts = [commonsense_prompt_sft.replace("<CONCEPT>", con) for con in concepts]

# print(concept_prompts)

file_name = "sft_dpo_finetunned_4bit_cs_prompt2_llama2_7b_properties_ufet_concepts.txt"

# response_list = []

with open(file_name, "w") as out_file:

    for concept_prompt in concept_prompts:

        sequences = pipeline(
            concept_prompt,
            do_sample=True,
            # top_p=,
            # top_k=,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=4000,
            max_new_tokens=1000,
            return_full_text=False,
            repetition_penalty=1.0,
            length_penalty=1.0,
        )

        for seq in sequences:
            # response_list.append(f"{seq['generated_text']}\n\n")

            concept = (
                concept_prompt[concept_prompt.find("Concept:") :]
                .replace("\n[/INST]", "")
                .replace("Concept:", "")
                .strip()
            )
            gen_props = seq["generated_text"].lstrip(", ")
            # split_prop = gen_props.lstrip(', ').split(", ")

            print(concept)
            print(gen_props)
            # print (type(split_prop))

            out_file.write(f"{concept}\t{gen_props}\n")

            print("===================================")

        del seq
        del sequences

del model
del pipeline
del concept_prompts

gc.collect()
gc.collect()
gc.collect()
