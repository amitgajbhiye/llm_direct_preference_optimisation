import os
import re
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

base_model = "meta-llama/Llama-2-7b-chat-hf"
inp_file = "data/ufet/clean_types.txt"  # Specify property file here
sft_adapter = "llama_finetuning_results/checkpoint-2900/"  # Model Prepration

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


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right"

model_vocab = set(tokenizer.get_vocab().keys())

with open(inp_file, "r") as inp_file:
    concepts = inp_file.readlines()

concepts = set([con.strip("\n").replace("_", " ").lower() for con in concepts])

common_concepts = concepts.intersection(model_vocab)
non_common_concepts = concepts.difference(model_vocab)
# non_common_concepts = [con for con in concepts if con not in model_vocab]

print("#" * 50)
print(f"Vocab: {len(model_vocab)}")
print(f"Number of concepts: {len(concepts)}")

print("#" * 50)
print(f"common_concepts: {len(common_concepts)}")
print(common_concepts)

print("#" * 50)
print(f"non_common_concepts: {len(non_common_concepts)}")
print(non_common_concepts)

print("#" * 50)


# pipeline = transformers.pipeline(
#     "text-generation", model=model, device_map="auto", tokenizer=tokenizer
# )


# with open(inp_file, "r") as inp_file:
#     concepts = inp_file.readlines()

# concepts = [con.strip("\n").replace("_", " ").lower() for con in concepts]

# print(f"Number of concepts: {len(concepts)}")


# basic_prompt = f"What is the most salient property of <CONCEPT>? Generate only the property and do not explain the property."

# commonsense_prompt_1 = """<s>[INST] <<SYS>>
# You are a contestant in the general knowledge quiz contest and always answer all kinds of common sense questions accurately.
# Please ensure that your responses are socially unbiased and positive in nature.
# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
# If you don't know the answer, please don't share false information.
# <</SYS>>
# Write the five most salient properties of the following concept. The propeties must be written in a Python list format. Limit the number of properties to 5. Concept: <CONCEPT>
# [/INST]"""


# # commonsense_prompt_2 = """<s>[INST] <<SYS>>
# # You are a contestant in the general knowledge quiz contest and always answer all kinds of common sense questions accurately. All output must be in valid JSON. Don't add explanation beyond the JSON.
# # Please ensure that your responses are socially unbiased and positive in nature.
# # If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
# # If you don't know the answer, please don't share false information.
# # <</SYS>>
# # Write the ten most salient properties of the following concept. Output must be in valid JSON like the following example {{"concept": concept, "properties": [in_less_than_ten_words]}}. Output must include only JSON.
# # All output must be in valid JSON. Don't add any explanations before and after the JSON.
# # Concept: <CONCEPT>
# # [/INST]"""


# commonsense_prompt_for_sft = """<s>[INST] <<SYS>>
# You are a contestant in the general knowledge quiz contest and always answer all kinds of commonsense questions accurately.
# All output must be in valid Python List. Don't add explanation beyond the Python List.
# If you don't know the answer, please don't share false information.
# <</SYS>>
# Write a property of the following concept.
# Output must be in valid a Python List like the following example [first property of the given concept in less than ten words, second property of the given concept in less than ten words,..., tenth property of the given concept in less than ten words].
# All output must be in a valid Python List. Don't add any explanations before and after the Python List.
# Concept: <CONCEPT>
# [/INST]"""


# commonsense_prompt_dpo_sft = """<s>[INST] <<SYS>>
# You are a contestant in the general knowledge quiz contest and always answer all kinds of common sense questions accurately.
# The output must be a valid property of the concept. Don't add explanation beyond the property.
# Please ensure that your responses are socially unbiased and positive in nature.
# If you don't know the answer, please don't share false information.
# <</SYS>>
# Write the most salient property of the following concept. The output must be a valid property of the concept. Don't add explanation beyond the property.
# Concept: <CONCEPT>
# [/INST]"""

# print(f"Prompt used is : {commonsense_prompt_dpo_sft}")

# concept_prompts = [
#     commonsense_prompt_dpo_sft.replace("<CONCEPT>", con) for con in concepts
# ]

# # print(concept_prompts)

# file_name = (
#     "dpo_sft_n_props_output_4bit_cs_prompt2_llama2_7b_properties_ufet_concepts.txt"
# )

# # response_list = []

# with open(file_name, "w") as out_file:

#     for concept_prompt in concept_prompts:
#         for i in range(10):
#             start_index = concept_prompt.find("Concept: ") + len("Concept: ")
#             end_index = concept_prompt.find("\n", start_index)
#             concept = concept_prompt[start_index:end_index].strip()

#             sequences = pipeline(
#                 concept_prompt,
#                 do_sample=True,
#                 # top_p=,
#                 # top_k=,
#                 num_return_sequences=1,
#                 eos_token_id=tokenizer.eos_token_id,
#                 max_length=4000,
#                 max_new_tokens=1000,
#                 return_full_text=False,
#                 repetition_penalty=1.0,
#                 length_penalty=1.0,
#             )

#             for seq in sequences:

#                 prop = str(seq["generated_text"]).lstrip("[").rstrip("]")

#                 print(f'{i}: {concept}:\t{prop.replace("[", "").replace("]", "")}\n')
#                 out_file.write(f'{concept}\t{prop.replace("[", "").replace("]", "")}\n')

#                 print("===================================")

#             del seq
#             del sequences

# del model
# del pipeline
# del concept_prompts


# gc.collect()
# gc.collect()
# gc.collect()
