import gc
import time

import pandas as pd
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

start_time = time.time()

base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
inp_file = "data/ontology_concepts/transport_vocab.txt"

wine_ontology_file = "data/ontology_concepts/wine/wine_vocab.txt"
olympics_ontology_file = "data/ontology_concepts/olympics/olympics_vocab.txt"
economics_ontology_file = "data/ontology_concepts/economy/economy_vocab.txt"

# Evaluation Taxonomy
food_taxo = "data/evaluation_taxo/terms/food.terms"
equipment_taxo = "data/evaluation_taxo/terms/equipment.terms"
science_taxo = "data/evaluation_taxo/terms/science.terms"
science_ev_taxo = "data/evaluation_taxo/terms/science_ev.terms"

df = pd.read_csv(science_ev_taxo, sep="\t", names=["id", "concept"])


# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load base moodel in quantised form
model = AutoModelForCausalLM.from_pretrained(
    base_model, quantization_config=bnb_config, device_map=0
)

print(f"############ Model ############", end="\n\n")
print(model, end="\n\n")
print(f"Device map")
print(model.hf_device_map)


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

pipeline = transformers.pipeline(
    "text-generation", model=model, device_map="auto", tokenizer=tokenizer
)

# with open(economics_ontology_file, "r") as inp_file:
#     concepts = inp_file.readlines()
# concepts = [con.strip("\n").replace("_", " ").lower() for con in concepts]

concepts = [con.strip() for con in df["concept"].values]

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
All output must include only valid JSON like the following example {"concept": concept, "facet_properties_dict": {facet: [list of properties with each property less than ten words long]}}.
Don't add any explanations before and after the JSON.
If you don't know the answer, please don't share false information.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

For the concept of the banana, write its different facets and most salient properties under each facet.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{"concept": "banana", "facet_properties_dict": {"category": ["fruit", "healthy snack", "food", etc.], "color": ["yellow", "green"], "shape": ["curved"], "taste": ["sweet"], "nutritional content": ["rich in potassium", "high in sugar", etc.], "used for": ["healthy snack", "making cakes", "making smoothie", etc.], "located at": ["banana tree", "supermarket", "fridge", "fruit bowl", etc.]}}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Write the ten most salient properties of the concept "boat".<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{"concept": "boat", "properties": ["located in bodies of water", "made of fiber", "means of transport", "used for fishing", "located in garages", "used for recreation", "used for rowing", "used for travel", "have motors", "have sails"]}<|eot_id|>
Write the ten most salient properties of the concept "cake".<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{"concept": "cake", "properties": ["deserts", "used for weddings", "used for special occasions", "made of eggs", "located in bakeries", "located in birthday parties", "made of flour", "located in grocery stores", "used for birthdays", "baked in ovens"]}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Write the ten most salient properties of the concept "<CONCEPT>".<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""


llama3_8B_1inc_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a contestant in the general knowledge quiz contest and always answer all kinds of common sense questions accurately. 
All output must include only valid JSON like the following example {"concept": concept, "facet_properties_dict": {facet: [list of properties with each property less than ten words long]}}.
Don't add any explanations before and after the JSON.
If you don't know the answer, please don't share false information.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

For the concept of the banana, write its different facets and most salient properties under each facet.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{"concept": "banana", "facet_properties_dict": {"category": ["fruit", "healthy snack", "food"], "color": ["yellow", "green"], "shape": ["curved"], "taste": ["sweet"], "nutritional content": ["rich in potassium", "high in sugar"], "used for": ["healthy snack", "making cakes", "making smoothie"], "located at": ["banana tree", "supermarket", "fridge", "fruit bowl"]}}<|eot_id|>

For the concept of the <CONCEPT>, write its different facets and most salient properties under each facet.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""


print(f"input_concepts: {concepts}")
print(f"Prompt used is : {llama3_8B_1inc_prompt}")
concept_prompts = [llama3_8B_1inc_prompt.replace("<CONCEPT>", con) for con in concepts]

repeat_times = 10
file_name = (
    f"llama3_repeat{repeat_times}_concept_facet_property_science_ev_onto_concepts.txt"
)

batch_size = 32
total_batches = len(concept_prompts) // batch_size


with open(file_name, "w") as out_file:

    for i in range(repeat_times):
        print(f"****** repeat_times : {i} ******")

        for batch_no, batch_start_idx in enumerate(
            range(0, len(concept_prompts), batch_size)
        ):
            print(f"****** processing_batch: {batch_no} / {total_batches} ******")

            concept_prompt_batch = concept_prompts[
                batch_start_idx : batch_start_idx + batch_size
            ]

            sequences = pipeline(
                concept_prompt_batch,
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
                print(f"{seq[0]['generated_text']}\n", flush=True)

                out_file.write(f'{seq[0]["generated_text"]}')

                print("===================================", flush=True)

            del seq
            del sequences

del model
del pipeline
del concept_prompts

end_time = time.time()
elapsed_time = end_time - start_time

hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = elapsed_time % 60

print(f"Execution time: {hours} hours, {minutes} minutes, and {seconds:.2f} seconds")

gc.collect()
gc.collect()
gc.collect()