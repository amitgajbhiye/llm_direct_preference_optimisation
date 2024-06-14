import gc
import json

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
All output must include only valid JSON like the following example {"concepts": list of concepts, "class": [class of concepts in less than ten words]}.
Don't add any explanations before and after the JSON.
If you don't know the answer, please don't share false information.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

You are an ontology engineer building a transport ontology. In the ontology what class would you assign to the following group of concepts <CONCEPT_LIST>.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""


llama3_8B_concepts_common_label_prompt_2 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an ontology engineer building a transport ontology.
All output must include only valid JSON like the following example {"class": [class of concepts from the list of classes]}.
Don't add any explanations before and after the JSON.
If you don't know the answer, please don't share false information.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

You are an ontology engineer building a transport ontology. From the list of classes: <LABEL_LIST>; assign a class label that best describes the group of concepts: <CONCEPT_LIST>.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""


prompt = 2

file_name = "llama3_common_label_clustered_concepts.txt"
print(f"Prompt used is : {llama3_8B_concepts_common_label_prompt}")

concepts_common_label = []

with open(file_name, "w") as out_file:

    for cl_label in cluster_labels:

        if prompt != 2:

            print(f"cluster_label:", cl_label)

            concepts_list = concept_cluster_labels[
                concept_cluster_labels["cluster_label"] == cl_label
            ]["concept"].unique()

            num_clustered_concepts = len(concepts_list)

            if num_clustered_concepts >= 10:
                prompt_concepts = concepts_list[:10]
            else:
                prompt_concepts = concepts_list

            prompt_concepts = ", ".join(prompt_concepts)

            concept_prompt = llama3_8B_concepts_common_label_prompt.replace(
                "<CONCEPT_LIST>", prompt_concepts
            )

            print(f"concepts_list:{concepts_list}")
            print(f"prompt_concepts:{prompt_concepts}")

            print(f"concept_prompt: {concept_prompt}")

        else:
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
            concept_prompt,
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
            # response_list.append(f"{seq['generated_text']}\n\n")
            print(f"{seq['generated_text']}\n")

            # out_file.write(f"\nprompt_concepts:{prompt_concepts}")
            out_file.write(f'{seq["generated_text"]}')

            # concept_facet_generated_data.append((concept, facet, seq["generated_text"]))

            print("===================================")

        del seq
        del sequences

del model
del pipeline

gc.collect()
gc.collect()
gc.collect()
