import gc

import pandas as pd
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

gc.collect()
gc.collect()

torch.cuda.empty_cache()
torch.cuda.empty_cache()

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

concept_facet_property_file = "data/ontology_concepts/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/facet_property/final_concept_facet_propert_clusterlabel.txt"

concept_facet_property_df = pd.read_csv(concept_facet_property_file, sep="\t")

concept_cluster_labels = concept_facet_property_df[["concept", "cluster_label"]]
cluster_labels = concept_cluster_labels["cluster_label"].unique()

print(f"input_df")
print(concept_facet_property_df)


llama3_8B_concepts_common_label_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an ontology engineer building a transport ontology.
Assign only one class to the group of concepts.
All output must include only valid JSON like the following example {"class": class of the concepts in less than ten words}.
Don't add any explanations before and after the JSON.
If you don't know the answer, please don't share false information.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

You are an ontology engineer building a transport ontology.
Assign only one class to the group of concepts.
In the ontology what class would you assign to the following group of concepts: <CONCEPT_LIST>.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""


llama3_8B_concepts_common_label_prompt_2 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an ontology engineer building a transport ontology.
Assign only one class to the group of concepts.
All output must include only valid JSON like the following example {"class": class of concepts from the list of classes}.
Don't add any explanations before and after the JSON.
If you don't know the answer, please don't share false information.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

You are an ontology engineer building a transport ontology. Assign only one class to the group of concepts.
From the list of classes: <LABEL_LIST>; assign a class label that best describes the group of concepts: <CONCEPT_LIST>.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""


llama3_8B_concepts_common_label_property_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an advanced language model with extensive knowledge of various concepts and their properties. Your task is to assign a concise, descriptive label to a given group of properties for a group of concept. The label should accurately capture the essence of the concepts based on the provided properties.
All output must include only valid JSON like the following example {"label": concise and descriptive label.}.
Don't add any explanations before and after the JSON.
If you don't know the answer, please don't share false information.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

You are an advanced language model with extensive knowledge of various concepts and their properties. Your task is to assign a concise, descriptive label to a given group of properties for a group of concept. The label should accurately capture the essence of the concepts based on the provided properties.

The properties are: <PROPERTY_LIST>.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""


prompt_type = "property_prompt"  # concept_prompt, concept_property_prompt

file_name = "llama3_clustered_concepts_common_label_from_only_property_label.txt"

print(f"prompt_type: {prompt_type}")
print(f"Prompt used is : {llama3_8B_concepts_common_label_property_prompt}")

concepts_common_label = []

with open(file_name, "w") as out_file:

    for cl_label in cluster_labels:

        if prompt_type == "concept_prompt":

            print(f"cluster_label:", cl_label)

            concepts_list = concept_cluster_labels[
                concept_cluster_labels["cluster_label"] == cl_label
            ]["concept"].unique()

            num_clustered_concepts = len(concepts_list)

            concept_prompt = llama3_8B_concepts_common_label_prompt.replace(
                "<CONCEPT_LIST>", str(concepts_list)
            )

            print(f"concepts_list:{concepts_list}")
            # print(f"prompt_concepts:{prompt_concepts}")

            print(f"concept_prompt: {concept_prompt}")

        elif prompt_type == "property_prompt":

            property_list = concept_facet_property_df[
                concept_facet_property_df["cluster_label"] == cl_label
            ]["property"].unique()

            property_list = ", ".join([prop.strip() for prop in property_list])

            num_clustered_properties = len(property_list)

            prop_prompt = llama3_8B_concepts_common_label_prompt.replace(
                "<PROPERTY_LIST>", str(property_list)
            )

            print(
                f"num_clustered_properties: {num_clustered_properties}, prop_prompt:{prop_prompt}"
            )

        elif prompt_type == "concept_property_prompt":
            concepts_df = concept_facet_property_df[
                concept_cluster_labels["cluster_label"] == cl_label
            ]

            print(f"concepts_df")
            print(concepts_df)

            cons = concepts_df["concept"].unique()
            props = concepts_df["property"].unique()

            concept_prompt = llama3_8B_concepts_common_label_prompt_2.replace(
                "<LABEL_LIST>", str(props)
            ).replace("<CONCEPT_LIST>", str(cons))

            print(f"concepts_list:{cons}")
            print(f"label_list:{props}")

            print(f"concept_prompt: {concept_prompt}")

        sequences = pipeline(
            prop_prompt,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=500,
            return_full_text=False,
            repetition_penalty=1.0,
            length_penalty=1.0,
            truncation=True,
            # max_length=3000,
            # top_p=,
            # top_k=,
        )

        for seq in sequences:

            print(f"\nProperties: {str(property_list)}")
            print(f"{seq['generated_text']}\n")

            out_file.write(f"\nProperties: {str(property_list)}")
            out_file.write(f'{seq["generated_text"]}\n')
            out_file.flush()

            # concept_facet_generated_data.append((concept, facet, seq["generated_text"]))

            print("===================================")

        del seq
        del sequences

del model
del pipeline

gc.collect()
gc.collect()
gc.collect()
