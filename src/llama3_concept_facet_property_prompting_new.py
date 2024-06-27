import gc
import json
import logging
import os
import time
from argparse import ArgumentParser

import pandas as pd
import torch
from huggingface_hub import login

# import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

device = 0
start_time = time.time()
access_token = "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"

login(token=access_token)


def get_execution_time(start_time, end_time):

    elapsed_time = end_time - start_time

    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    print(
        f"Execution time: {hours} hours, {minutes} minutes, and {seconds:.2f} seconds",
        flush=True,
    )

    logger.info(
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


def prepare_data(config):

    input_file = config["input_file"]

    # with open(economics_ontology_file, "r") as inp_file:
    #     concepts = inp_file.readlines()
    # concepts = [con.strip("\n").replace("_", " ").lower() for con in concepts]

    df = pd.read_csv(input_file, sep="\t", names=["id", "concept"])
    concepts = [con.strip() for con in df["concept"].values]

    print(f"Number of concepts: {len(concepts)}")
    print(f"input_concepts: {concepts}")
    print(f"Prompt used is : {llama3_8B_1inc_prompt}")

    logger.info(f"Number of concepts: {len(concepts)}")
    logger.info(f"input_concepts: {concepts}")
    logger.info(f"Prompt used is : {llama3_8B_1inc_prompt}")

    concept_prompts = [
        llama3_8B_1inc_prompt.replace("<CONCEPT>", con) for con in concepts
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
    )

    print(f"base_model: {base_model}")
    print(f"############ Model ############", end="\n\n")
    print(model, end="\n\n")
    print(f"Device map")
    print(model.hf_device_map)

    logger.info(f"base_model: {base_model}")
    logger.info(f"############ Model ############")
    logger.info(model)
    logger.info(f"Device map: {model.hf_device_map}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    generator = pipeline(
        "text-generation", model=model, device_map=device, tokenizer=tokenizer
    )

    repeat_times = config["repeat_times"]
    output_file = config["output_file"]
    batch_size = config["batch_size"]

    total_batches = len(concept_prompts) // batch_size

    with open(output_file, "w") as out_file:

        for i in range(repeat_times):
            print(f"****** repeat_times : {i} ******")

            for batch_no, batch_start_idx in enumerate(
                range(0, len(concept_prompts), batch_size)
            ):
                print(f"****** processing_batch: {batch_no} / {total_batches} ******")
                logger.info(
                    f"****** processing_batch: {batch_no} / {total_batches} ******"
                )

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
                    print(f"{seq[0]['generated_text']}\n", flush=True)

                    out_file.write(f'{seq[0]["generated_text"]}')
                    # logger.info(f"{seq[0]['generated_text']}\n")

                    print("===================================", flush=True)

                del seq
                del sequences
                end_time = time.time()
                print(
                    f"batch_processing_time: {get_execution_time(start_time, end_time)}"
                )

    del model
    del pipeline
    del concept_prompts

    end_time = time.time()
    get_execution_time(start_time, end_time)

    gc.collect()
    gc.collect()
    gc.collect()


if __name__ == "__main__":

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
    generate_data(config, concept_prompts)

    logger.info(f"Job Finished")
