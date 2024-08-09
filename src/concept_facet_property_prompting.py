import gc
import json
import logging
import math
import os
import time
from argparse import ArgumentParser

import pandas as pd
import torch
import transformers
from huggingface_hub import login
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

device = 0
access_token = "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"

login(token=access_token)


def get_execution_time(start_time, end_time):

    elapsed_time = end_time - start_time

    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    # print(
    #     f"Execution time: {hours} hours, {minutes} minutes, and {seconds:.2f} seconds",
    #     flush=True,
    # )

    return (
        f"Execution time: {hours} hours, {minutes} minutes, and {seconds:.2f} seconds"
    )


def read_config(config_file):
    with open(config_file, "r") as json_file:
        config_dict = json.load(json_file)

    return config_dict


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


###########
llama3_8B_5inc_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a contestant in the general knowledge quiz contest and always answer all kinds of common sense questions accurately. 
All output must include only valid JSON like the following example {"concept": concept, "facet_properties_dict": {facet: [list of properties with each property less than ten words long]}}.
Don't add any explanations before and after the JSON.
If you don't know the answer, please don't share false information.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

For the concept of teacher, write its different facets and most salient properties under each facet.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{"concept": "teacher", "facet_properties_dict": {"role": ["teaching", "guiding students", "assessing", "classroom management"], "workplace": ["school", "college", "university", "training centre"], "skill": ["creativity", "communication", "patience", "subject expertise", "organisation"], "tool": ["textbook", "whiteboard", "computer", "educational materials"], "impact": ["knowledge transfer", "skill development", "inspiration", "career guidance"], "types": ["primary school teacher", "high school teacher", "college professor", "special education teacher"]}}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

For the concept of bamboo, write its different facets and most salient properties under each facet.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{"concept": "bamboo", "facet_properties_dict": {"category": ["plant", "grass family", "natural resource"], "growth": ["fast-growing", "tall", "woody stems"], "used for": ["construction", "furniture", "paper", "textiles", "food for pandas"], "location": ["Asia", "tropical regions", "forests", "gardens"], "appearance": ["green stem", "hollow", "jointed nodes"], "environmental impact": ["afforestation", "carbon sequestration", "erosion control", "renewable resource"], "types": ["clumping bamboo", "running bamboo"]}}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

For the concept of Bill Gates, write its different facets and most salient properties under each facet.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{"concept": "Bill Gates", "facet_properties_dict": {"profession": ["entrepreneur", "philanthropist", "author"], "founded": ["Microsoft", "Bill & Melinda Gates Foundation"], "known for": ["co-founding Microsoft", "Windows software", "philanthropy"], "born": ["October 28, 1955"], "philanthropy": ["global health", "education", "poverty alleviation", "climate change"], "publications": ["The Road Ahead", "Business @ the Speed of Thought", "How to Avoid a Climate Disaster", "How to Prevent the Next Pandemic"], "awards": ["Presidential Medal of Freedom", "Padma Bhushan", "Knight Commander of the Order of the British Empire"]}}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

For the concept of human, write its different facets and most salient properties under each facet.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{"concept": "human", "facet_properties_dict": {"anatomy": ["brain", "heart", "lungs", "skin", "muscles", "bones"], "characteristics": ["bipedal", "intelligent", "social", "emotional", "tool users"], "life stages": ["infant", "child", "adolescent", "adult", "elderly"], "used for": ["social interaction", "cultural development", "technology creation", "reproduction"], "location": ["earth", "cities", "villages", "homes"], "needs": ["food", "water", "shelter", "community", "education"], "communication": ["language", "gestures", "writing", "art"]}}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

For the concept of book, write its different facets and most salient properties under each facet.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{"concept": "book", "facet_properties_dict": {"types": ["fiction", "non-fiction", "textbook", "manual", "graphic novel"], "parts": ["cover", "pages", "spine", "chapters", "table of contents"], "used for": ["reading", "education", "entertainment", "reference"], "formats": ["physical", "ebook", "audiobook"], "materials": ["paper", "ink", "binding materials"], "location": ["library", "bookstore", "home", "school", "college"]}}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

For the concept of <CONCEPT>, write its different facets and most salient properties under each facet.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""


def prepare_data(config):

    input_file = config["input_file"]

    # with open(economics_ontology_file, "r") as inp_file:
    #     concepts = inp_file.readlines()
    # concepts = [con.strip("\n").replace("_", " ").lower() for con in concepts]

    df = pd.read_csv(input_file, sep="\t", names=["id", "concept"])
    concepts = [con.strip() for con in df["concept"].unique()]

    logger.info(f"Number of concepts: {len(concepts)}")
    logger.info(f"input_concepts: {concepts}")
    logger.info(f"Prompt used is : {llama3_8B_5inc_prompt}")

    concept_prompts = [
        llama3_8B_5inc_prompt.replace("<CONCEPT>", con) for con in concepts
    ]

    return concept_prompts


def generate_data(config, concept_prompts):

    base_model = config["base_model"]

    # Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base moodel in quantised form
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=device,
        token=access_token,
        cache_dir="hf_model_cache/",
    )

    model_config = AutoConfig.from_pretrained(base_model)

    logger.info(f"base_model: {base_model}")
    logger.info(f"############ Model ############")
    logger.info(model)
    logger.info(f"Device map: {model.hf_device_map}")
    logger.info(f"model_context_length: {model_config.max_position_embeddings}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    generator = transformers.pipeline(
        "text-generation", model=model, device_map=device, tokenizer=tokenizer
    )

    repeat_times = config["repeat_times"] + 1
    output_file = os.path.join(config["output_dir"], config["output_file"])
    batch_size = config["batch_size"]

    total_batches = math.ceil(len(concept_prompts) / batch_size)

    with open(output_file, "w") as out_file:

        for i in range(1, repeat_times):

            for batch_no, batch_start_idx in enumerate(
                range(0, len(concept_prompts), batch_size), start=1
            ):
                batch_start_time = time.time()

                concept_prompt_batch = concept_prompts[
                    batch_start_idx : batch_start_idx + batch_size
                ]

                sequences = generator(
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

                    out_file.write(f'{seq[0]["generated_text"]}')
                    out_file.flush()

                    # print(f"{seq[0]['generated_text']}\n", flush=True)
                    # print("===================================", flush=True)

                del seq
                del sequences
                batch_end_time = time.time()

                print(
                    f"repeat_times: {i}, processed_batch: {batch_no} / {total_batches}, batch_size: {batch_size}, {get_execution_time(batch_start_time, batch_end_time)}",
                    flush=True,
                )

                logger.info(
                    f"repeat_times: {i}, processed_batch: {batch_no} / {total_batches}, batch_size: {batch_size}, {get_execution_time(batch_start_time, batch_end_time)}"
                )

    del model
    del generator
    del concept_prompts

    gc.collect()
    gc.collect()
    gc.collect()

    logger.info(f"Finished generating data from LLM")
    logger.info(f"The raw output from LLM is saved at: {output_file}")
    return output_file


def parse_and_format_data(file_path, config):

    logger.info(f"Parsing the LLM data: {file_path}")

    concepts = []
    with open(file_path, "r") as file:
        for line in file:

            # if not line.endswith("}}"):
            #     line = line + "}"

            if line.strip():
                try:
                    data = json.loads(line.strip())
                    concept = data.get("concept")
                    facet_properties_dict = data.get("facet_properties_dict")

                    if concept and facet_properties_dict:
                        concepts.append(
                            {
                                "concept": concept,
                                "facet_properties_dict": facet_properties_dict,
                            }
                        )
                except json.JSONDecodeError as e:
                    print(f"JSON decoding error: {e}")
                    print(f"Problematic line: {line}")
                    logger.info(f"JSON decoding error: {e}")
                    logger.info(f"Problematic line: {line}")

    df_fixed = pd.DataFrame(concepts)

    concept_facet_property = []

    for idx, row in df_fixed.iterrows():

        concept = row["concept"].lower().strip()
        facet_properties_dict = row["facet_properties_dict"]
        # facets = facet_properties_dict.keys()

        for facet, properties in facet_properties_dict.items():
            for prop in properties:
                concept_facet_property.append(
                    (concept, facet.lower().strip(), prop.lower().strip())
                )

    df = pd.DataFrame.from_records(
        concept_facet_property, columns=["concept", "facet", "property"]
    )
    logger.info(f"all_records: {df.shape}")

    df.drop_duplicates(inplace=True)
    logger.info(f"after_drop_duplicate: {df.shape}")

    df["facet_property"] = df["facet"].str.strip() + ": " + df["property"].str.strip()
    df.sort_values(by="facet", inplace=True)

    file_name, file_extension = os.path.splitext(config["output_file"])

    parsed_file_name = file_name + "_parsed.txt"
    parsed_file = os.path.join(config["output_dir"], parsed_file_name)

    df.to_csv(parsed_file, sep="\t", index=False)

    colon_file_name = file_name + "_facet_colon_property.txt"

    colon_file = os.path.join(config["output_dir"], colon_file_name)
    os.path.join(config["output_dir"], parsed_file_name)

    df["facet_property"].drop_duplicates().to_csv(
        colon_file, sep="\t", index=0, header=0
    )

    logger.info(f"Parsed - concept_facet_property file saved at: {parsed_file_name}")
    logger.info(f"facet_colon_property file saved at: {colon_file}")

    return df


if __name__ == "__main__":
    start_time = time.time()

    parser = ArgumentParser(description="Fine tune configuration")

    parser.add_argument(
        "--config_file",
        default="configs/default_config.json",
        help="path to the configuration file",
    )
    args = parser.parse_args()

    config = read_config(args.config_file)

    log_dir = config["log_dir"]
    log_file_name = os.path.join(
        log_dir,
        f"log_{config['experiment_name']}_{time.strftime('%d-%m-%Y_%H-%M-%S')}.txt",
    )
    logging.basicConfig(
        level=logging.INFO,
        filename=log_file_name,
        filemode="w",
        format="%(asctime)s : %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Reading Configuration File: {args.config_file}")

    logger.info("The model is run with the following configuration")
    logger.info(f"\n {config} \n")

    # Executing pipleine
    concept_prompts = prepare_data(config)
    concept_facet_property_file = generate_data(config, concept_prompts)

    parse_and_format_data(file_path=concept_facet_property_file, config=config)

    logger.info(f"job_finished")
    end_time = time.time()
    logger.info(f"total_execution_time: {get_execution_time(start_time, end_time)})")
