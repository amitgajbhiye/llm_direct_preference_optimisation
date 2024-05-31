import gc
import json
import os

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

base_model = "meta-llama/Meta-Llama-3-8B-Instruct"


# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load base moodel in quantised form
model = AutoModelForCausalLM.from_pretrained(
    base_model, quantization_config=bnb_config, device_map={"": 0}
)

print(f"############ Model ############", end="\n\n")
print(model, end="\n\n")


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

pipeline = transformers.pipeline(
    "text-generation", model=model, device_map="auto", tokenizer=tokenizer
)


def parse_concepts_categories(file_path):
    results = []

    with open(file_path, "r") as file:
        for line in file:
            try:
                # Load the JSON line
                record = json.loads(line)
                concepts = record["concepts"]
                categories = record["category"]

                for concept in concepts:
                    for category in categories:
                        results.append((concept, category))

            except json.JSONDecodeError:
                print(f"Failed to parse line: {line}")

    return results


file_path = "../llm_direct_preference_optimisation/data/ontology_concepts/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/llama3_with_3inc_exp_generated_transport_concepts_group_five_categories.txt"

concepts_categories = parse_concepts_categories(file_path)


llama3_8B_category_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a contestant in the general knowledge quiz contest and always answer all kinds of common sense questions accurately. 
All output must include only valid JSON like the following example {"concept": concept, "properties": [in_less_than_ten_words]}.
Don't add any explanations before and after the JSON.
If you don't know the answer, please don't share false information.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

In terms of <CATEGORY>, write the ten most salient properties of the concept <CONCEPT>.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""


# file_name = f"{base_model.replace("-", "_").replace("/", "_")}_generated_inc_ex__concepts_properties.txt".lower()

file_name = "llama3_transport_category_based_concepts_properties.txt"

print(f"Prompt used is : {llama3_8B_category_prompt}")

concept_category_prompts = [
    llama3_8B_category_prompt.replace("<CATEGORY>", cat).replace("<CONCEPT>", con)
    for con, cat in concepts_categories
]

with open(file_name, "w") as out_file:
    for con, cat in concepts_categories:
        concept_prompt = llama3_8B_category_prompt.replace("<CATEGORY>", cat).replace(
            "<CONCEPT>", con
        )

        print(f"concept_prompt: {concept_prompt}")
        sequences = pipeline(
            concept_prompt,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=500,
            return_full_text=False,
            repetition_penalty=1.0,
            length_penalty=1.0,
            truncation=True,
            # max_length=500,
            # top_p=,
            # top_k=,
        )

        for seq in sequences:
            # response_list.append(f"{seq['generated_text']}\n\n")

            print(f"Category: {cat}\n")
            print(f"{seq['generated_text']}\n")

            out_file.write(f"Category: {cat}")
            out_file.write(f'{seq["generated_text"]}')

            print("===================================")

        del seq
        del sequences


# with open(file_name, "w") as out_file:
#     for concept_prompt in concept_category_prompts:
#         print(f"concept_prompt: {concept_prompt}")
#         sequences = pipeline(
#             concept_prompt,
#             do_sample=True,
#             num_return_sequences=1,
#             eos_token_id=tokenizer.eos_token_id,
#             max_new_tokens=500,
#             return_full_text=False,
#             repetition_penalty=1.0,
#             length_penalty=1.0,
#             truncation=True,
#             # max_length=500,
#             # top_p=,
#             # top_k=,
#         )

#         for seq in sequences:
#             # response_list.append(f"{seq['generated_text']}\n\n")
#             print(f"{seq['generated_text']}\n")

#             out_file.write(f'{seq["generated_text"]}')

#             print("===================================")

#         del seq
#         del sequences

del model
del pipeline

gc.collect()
gc.collect()
gc.collect()
