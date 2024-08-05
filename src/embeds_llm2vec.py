import gc
import os
import pickle

import torch
from llm2vec import LLM2Vec
from peft import PeftModel
from transformers import AutoConfig, AutoModel, AutoTokenizer

torch.cuda.empty_cache()
torch.cuda.empty_cache()

gc.collect()
gc.collect()
gc.collect()


def prepare_model(MODEL_ID):

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )

    model = PeftModel.from_pretrained(
        model,
        MODEL_ID,
    )

    llm2vec_model = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)

    return llm2vec_model


# MODEL_ID = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"

MODEL_ID = "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"

llm2vec_model = prepare_model(MODEL_ID=MODEL_ID)


# facet_colon_property_files = [
#     "data/evaluation_taxo/generated_facet_property/llama3_repeat10_food_facet_colon_property.txt",
#     "data/evaluation_taxo/generated_facet_property/llama3_repeat10_science_facet_colon_property.txt",
#     "data/evaluation_taxo/generated_facet_property/llama3_repeat10_equipment_facet_colon_property.txt",
#     "data/evaluation_taxo/generated_facet_property/llama3_repeat10_commonsense_facet_colon_property.txt",
#     "data/evaluation_taxo/generated_facet_property/llama3_repeat10_environment_facet_colon_property.txt",
# ]


facet_colon_property_files = [
    "data/evaluation_taxo/generated_facet_property/llama3_commonsense_property.tsv",
    "data/evaluation_taxo/generated_facet_property/llama3_environment_property.tsv",
    "data/evaluation_taxo/generated_facet_property/llama3_equipment_property.tsv",
    "data/evaluation_taxo/generated_facet_property/llama3_food_property.tsv",
    "data/evaluation_taxo/generated_facet_property/llama3_science_property.tsv",
]

for fact_property_file in facet_colon_property_files:
    print(f"getting_embeddings: {fact_property_file}", flush=True)

    with open(fact_property_file, "r") as fin:
        facet_property = [fp.strip("\n") for fp in fin.readlines()]

    print(f"num_facet_property: {len(facet_property)}")

    facet_property_embeds = llm2vec_model.encode(facet_property).detach().cpu().numpy()

    print(f"facet_property_embeds.shape: {facet_property_embeds.shape}")

    facet_property_and_embedding = [
        (facet_prop, embed)
        for facet_prop, embed in zip(facet_property, facet_property_embeds)
    ]

    facet_property_and_embedding_dict = {
        facet_prop: embed
        for facet_prop, embed in zip(facet_property, facet_property_embeds)
    }

    print(f"Top 5 facet_property_and_embedding")
    print(facet_property_and_embedding[0:5])

    # output_dir_path = os.path.dirname(fact_property_file)
    file_name_with_ext = os.path.basename(fact_property_file)
    file_name, file_extension = os.path.splitext(file_name_with_ext)

    out_file_name = os.path.join(os.path.basename(MODEL_ID), f"{file_name}_embeds.pkl")
    pickle_output_file = os.path.join("embeds", out_file_name)

    with open(pickle_output_file, "wb") as pkl_file:
        pickle.dump(facet_property_and_embedding_dict, pkl_file)

    print(f"got_embeddings: {fact_property_file}", flush=True)
    print(f"embeds_saved: {pickle_output_file}", flush=True)
    print(flush=True)

torch.cuda.empty_cache()
torch.cuda.empty_cache()

gc.collect()
gc.collect()
gc.collect()
