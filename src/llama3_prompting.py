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

base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
inp_file = "data/ontology_concepts/transport_vocab.txt"

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


with open(inp_file, "r") as inp_file:
    concepts = inp_file.readlines()
concepts = [con.strip("\n").replace("_", " ").lower() for con in concepts]

print(f"Number of concepts: {len(concepts)}")

# llama3_8B_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are a contestant in the general knowledge quiz contest and always answer all kinds of common sense questions accurately.
# All output must be in valid JSON. Don't add explanation beyond the JSON.
# Please ensure that your responses are socially unbiased and positive in nature.
# If you don't know the answer, please don't share false information.<|eot_id|>
# <|start_header_id|>user<|end_header_id|>
# Write the ten most salient properties of the following concept.
# Output must be in valid JSON like the following example {"concept": concept, "properties": [in_less_than_ten_words]}.
# All output must include only valid JSON.
# Don't add any explanations before and after the JSON.
# Concept: <CONCEPT> <|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>"""


llama3_8B_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a contestant in the general knowledge quiz contest and always answer all kinds of common sense questions accurately. 
All output must be in valid JSON. Don't add explanation beyond the JSON.
If you don't know the answer, please don't share false information.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Write the ten most salient properties of the following concept.
All output must include only valid JSON like the following example {"concept": concept, "properties": [in_less_than_ten_words]}.
Don't add any explanations before and after the JSON.
Concept: <CONCEPT> <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""


llama3_8B_inc_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a contestant in the general knowledge quiz contest and always answer all kinds of common sense questions accurately. 
All output must include only valid JSON like the following example {"concept": concept, "properties": [in_less_than_ten_words]}.
Don't add any explanations before and after the JSON.
If you don't know the answer, please don't share false information.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Write the ten most salient properties of the concept "acorn".<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{"concept": "acorn", "properties": ["fruit", "located in trees", "part of oak", "have seeds"]}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Write the ten most salient properties of the concept "action figure".<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{"concept": "action figure", "properties": ["doll", "located in toy stores", "used for playing"]}<|eot_id|>
Write the ten most salient properties of the concept "airplane".<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{"concept": "airplane", "properties": ["a large machine with wings", "have propellers", "used for traveling"]}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Write the ten most salient properties of the concept "acne medication".<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{"concept": "acne medication", "properties": ["used for getting rid of pimples", "located in drug stores", "medicine"]}<|end_header_id|
Write the ten most salient properties of the concept "animals".<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{"concept": "animals", "properties": ["part of ecosystems", "located in pet stores", "have bodies", "used for food"]}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Write the ten most salient properties of the concept "shed".<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{"concept": "shed", "properties": ["located in backyard", "used for storage", "have tools"]}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Write the ten most salient properties of the concept "<CONCEPT>".<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

llama3_8B_3inc_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a contestant in the general knowledge quiz contest and always answer all kinds of common sense questions accurately. 
All output must include only valid JSON like the following example {"concept": concept, "properties": [in_less_than_ten_words]}.
Don't add any explanations before and after the JSON.
If you don't know the answer, please don't share false information.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Write the ten most salient properties of the concept "bike".<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{"concept": "bike", "properties": ['used for riding', 'bicycle', 'capable of crashs', 'capable of skids', 'located in garages', 'located in streets', 'located in toy stores', 'used for commuting', 'used for transport', 'common in asia']}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Write the ten most salient properties of the concept "boat".<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{"concept": "boat", "properties": ['located in bodies of water', 'made of fiber', 'means of transport', 'used for fishing', 'located in garages', 'used for recreation', 'used for rowing', 'used for travel', 'have motors', 'have sails']<|eot_id|>
Write the ten most salient properties of the concept "cake".<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{"concept": "cake", "properties": ['deserts', 'used for weddings', 'used for special occasions', 'made of eggs', 'located in bakeries', 'located in birthday parties', 'made of flour', 'located in grocery stores', 'used for birthdays', 'baked in ovens']}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Write the ten most salient properties of the concept "<CONCEPT>".<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""


# file_name = f"{base_model.replace("-", "_").replace("/", "_")}_generated_inc_ex__concepts_properties.txt".lower()

file_name = "llama3_with_3inc_exp_generated_transport_concepts_properties.txt"

print(f"Prompt used is : {llama3_8B_inc_prompt}")

concept_prompts = [llama3_8B_inc_prompt.replace("<CONCEPT>", con) for con in concepts]

with open(file_name, "w") as out_file:
    for concept_prompt in concept_prompts:
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
            print(f"{seq['generated_text']}\n")

            out_file.write(f'{seq["generated_text"]}')

            print("===================================")

        del seq
        del sequences

del model
del pipeline
del concept_prompts


gc.collect()
gc.collect()
gc.collect()
