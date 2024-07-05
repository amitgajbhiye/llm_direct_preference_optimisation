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

concept_facet_property_file = "data/ontology_concepts/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/facet_property/llama3_concept_facet_property_transport_onto_concepts_parsed.txt"

with_inc_exp = False

df = pd.read_csv(concept_facet_property_file, sep="\t")

if with_inc_exp:
    df = df[["concept", "facet"]].drop_duplicates()

print(f"input_df")
print(df)


llama3_8B_3inc_concept_facet_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a contestant in the general knowledge quiz contest and always answer all kinds of common sense questions accurately. 
All output must include only valid JSON like the following example {"concept": concept, "properties": [list of ten properties with each property less than ten words long]}.
Don't add any explanations before and after the JSON.
If you don't know the answer, please don't share false information.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

In terms of the uses, write the ten most salient properties of the concept "petroleum product".<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{"concept": "petroleum product", "properties": ["fuels cars", "powers airplanes", "makes plastic", "heats homes", "electricity generation", "lubricants", "plastics production", "household products","asphalt and road Construction", "pharmaceuticals"]}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

In terms of the purpose, write the ten most salient properties of the concept "military watercraft".<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{"concept": "military watercraft", "properties": ["defend", "patrol", "attack", "transport", "protect", "rescue", "warfare", "humanitarian assistance", "search", "logistics"]}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

In terms of the location, write the ten most salient properties of the concept "fishing vessel".<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{"concept": "fishing vessel", "properties": ["sea", "lake", "river", "port", "ocean", "coastal waters", "fishing ports", "marinas", "fishing grounds", "reefs"]}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

In terms of the <FACET>, write the ten most salient properties of the concept "<CONCEPT>".<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""


llama3_8B_without_inc_exp_concept_facet_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a contestant in the general knowledge quiz contest and always answer all kinds of common sense questions accurately. 
All output must include only valid JSON like the following example {"concept": concept, "properties": [list of ten properties with each property less than ten words long]}.
The output properties must be different than the properties mentioned in the prompt.
Don't add any explanations before and after the JSON.
If you don't know the answer, please don't share false information.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

In terms of <FACET>, some of the properties of the concept "<CONCEPT>" are <PROPERTY_LIST>. Write ten such properties of the concept "<CONCEPT>".<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""


file_name = (
    "llama3_without_inc_exp_concept_facet_transport_onto_concepts_properties.txt"
)


print(f"Prompt used is : {llama3_8B_without_inc_exp_concept_facet_prompt}")
# concept_prompts = [llama3_8B_1inc_prompt.replace("<CONCEPT>", con) for con in concepts]

concept_facet_generated_data = []

with open(file_name, "w") as out_file:

    if with_inc_exp:

        for concept, facet in df.values:

            concept_prompt = llama3_8B_3inc_concept_facet_prompt.replace(
                "<FACET>", facet
            ).replace("<CONCEPT>", concept)

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
                print(f"\n\nfacet:{facet}")
                print(f"{seq['generated_text']}\n")

                out_file.write(f"\n\nfacet:{facet}")
                out_file.write(f'{seq["generated_text"]}')

                concept_facet_generated_data.append(
                    (concept, facet, seq["generated_text"])
                )

                print("===================================")

            del seq
            del sequences
    else:
        for con, fac in df[["concept", "facet"]].values:

            properties = df[(df["concept"] == con) & (df["facet"] == fac)][
                "property"
            ].to_list()

            prop_string = ", ".join(properties)

            print("***** con, fac, prop_string *****")
            print(con, "##", fac, "##", prop_string)

            concept_prompt = (
                llama3_8B_without_inc_exp_concept_facet_prompt.replace(
                    "<FACET>", fac, 2
                )
                .replace("<CONCEPT>", con, 2)
                .replace("<PROPERTY_LIST>", prop_string, 2)
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
                print(f"\n\nfacet:{fac}")
                print(f"{seq['generated_text']}\n")

                out_file.write(f"\n\nfacet:{fac}")
                out_file.write(f'{seq["generated_text"]}')

                concept_facet_generated_data.append((con, fac, seq["generated_text"]))

                print("===================================")

            del seq
            del sequences

# gen_df = pd.DataFrame(
#     concept_facet_generated_data, columns=["concept", "facet", "generated_data"]
# )
# gen_df.to_csv("concept_facet_generated_data.txt", sep="\t", index=False)


del model
del pipeline


gc.collect()
gc.collect()
gc.collect()