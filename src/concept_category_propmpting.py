import gc
import os

import pandas as pd
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
inp_file = "data/ontology_concepts/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/eps0_5_minsam3_transport_con_valid_cluster_llama38b_embeds.txt"


def create_concept_group(inp_file):

    inp_df = pd.read_csv(inp_file, sep="\t")

    print("inp_df")
    print(inp_df)

    unique_clusters_labels = inp_df["cluster_label"].unique()

    print("unique_clusters_labels")
    print(unique_clusters_labels)

    concept_groups = []
    for cluster_label in unique_clusters_labels:
        concepts_cluster = inp_df[inp_df["cluster_label"] == cluster_label][
            "concept"
        ].to_list()[:]
        concepts_cluster = ", ".join(concepts_cluster)

        print(f"concepts_cluster: {concepts_cluster}")
        concept_groups.append(concepts_cluster)

    return concept_groups


concept_group = create_concept_group(inp_file)

print(f"concept_group: {concept_group}")


llama3_8B_3inc_con_group_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a contestant in the general knowledge quiz contest and always answer all kinds of common sense questions accurately. 
All output must include only valid JSON like the following example {"concepts": [list of concepts to categorise], "category": [in_less_than_ten_words]}.
Don't add any explanations before and after the JSON.
If you don't know the answer, please don't share false information.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Write a category for the following group of concepts bench, stool, chair, desk, bookshelf.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{"concepts": ["bench", "stool", "chair", "desk", "bookshelf"], "category": ["furniture"]}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Write a category for the following group of concepts chessboard, cards, tennis, football.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{"concepts": ["chessboard", "cards", "tennis", "football"], "category": ["games"]}<|eot_id|>
Write the ten most salient properties of the concept "cake".<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{"concepts": ["whisk", "mortar", "kitchen shears"], "category": ["tools"]}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Write a category for the following group of concepts <CONCEPT_LIST>.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""


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


file_name = "llama3_with_3inc_exp_generated_transport_concepts_group_categories.txt"

print(f"Prompt used is : {llama3_8B_3inc_con_group_prompt}")
# concept_prompts = [llama3_8B_3inc_prompt.replace("<CONCEPT>", con) for con in concepts]

con_group_prompt = [
    llama3_8B_3inc_con_group_prompt.replace("<CONCEPT_LIST>", group)
    for group in concept_group
]

with open(file_name, "w") as out_file:
    for concept_prompt in con_group_prompt:
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
del con_group_prompt


gc.collect()
gc.collect()
gc.collect()
