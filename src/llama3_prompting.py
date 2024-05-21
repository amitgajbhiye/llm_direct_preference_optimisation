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

print(f"Prompt used is : {llama3_8B_prompt}")

concept_prompts = [llama3_8B_prompt.replace("<CONCEPT>", con) for con in concepts]
file_name = f"{base_model.replace("-", "_").replace("/", "_")}_generated_ueft_concepts_properties.txt".lower()


with open(file_name, "w") as out_file:
    for concept_prompt in concept_prompts:
        print (f"concept_prompt: {concept_prompt}")
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
