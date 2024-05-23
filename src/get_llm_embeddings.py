import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from llm2vec import LLM2Vec
from peft import PeftModel
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Loading base Mistral model, along with custom code that enables bidirectional connections in decoder-only LLMs. MNTP LoRA weights are merged into the base model.
tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp")
config = AutoConfig.from_pretrained(
    "McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp", trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp",
    trust_remote_code=True,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)
model = PeftModel.from_pretrained(
    model,
    "McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp",
)
model = model.merge_and_unload()  # This can take several minutes on cpu

# Loading supervised model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + supervised (LoRA).
model = PeftModel.from_pretrained(
    model, "McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-supervised"
)

# Wrapper for encoding and pooling operations
l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)

# Encoding documents. Instruction are not required for documents

con_prop_file = "data/ontology_concepts/llama3_with_3inc_exp_con_prop_parsed.txt"
con_prop_df = pd.read_csv(con_prop_file, sep="\t", names=["concept", "property"])

uniq_props = list(con_prop_df["property"].unique())

print(f"Num Unique Property: {len(uniq_props)}")
print(f"Unique Property")
# print(uniq_props)

llm_prop_embeds = l2v.encode(uniq_props).detach().cpu().numpy()

print(f"llm_prop_embeds.shape: {llm_prop_embeds.shape}")

props_and_embedding = [
    (prop, embed) for prop, embed in zip(uniq_props, llm_prop_embeds)
]

print(f"Top 5 props")
print(props_and_embedding[0:5])

pickle_output_file = "output_llm_embeds/llama3_with_3inc_exp_prop_embed.pkl"
with open(pickle_output_file, "wb") as pkl_file:
    pickle.dump(props_and_embedding, pkl_file)


# Assuming 'embeddings' is your array of property embeddings and 'properties' is a list of corresponding property names

# Normalize the embeddings
scaler = StandardScaler()
embeddings_normalized = scaler.fit_transform(llm_prop_embeds)

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5, metric="cosine", algorithm="brute")
clusters = dbscan.fit_predict(embeddings_normalized)

# Create a dictionary to map properties to their cluster labels
property_cluster_map = {
    property_name: cluster_label
    for property_name, cluster_label in zip(uniq_props, clusters)
}

print("property_cluster_map")
print(property_cluster_map)


prop_cluster_list = []
for property_name, cluster_label in property_cluster_map.items():
    prop_cluster_list.append((property_name, cluster_label))

print("prop_cluster_list")
print(prop_cluster_list)

property_cluster_df = pd.DataFrame.from_records(
    prop_cluster_list, columns=["property", "cluster_label"]
)

# property_cluster_df = pd.DataFrame.from_dict(property_cluster_map)
property_cluster_df.sort_values(by="cluster_label", inplace=True, ascending=False)

print(f"property_cluster_df")
print(property_cluster_df)

property_cluster_df.to_csv("property_cluster_df.txt", sep="\t", index=None)

con_prop_cluster_df = con_prop_df.copy(deep=True)


def get_prop_cluster_label(prop):
    return property_cluster_map[prop]


con_prop_cluster_df["cluster_label"] = [
    get_prop_cluster_label(p) for p in con_prop_cluster_df["property"]
]

con_prop_cluster_df.sort_values(by="cluster_label", inplace=True, ascending=False)

con_prop_cluster_df.to_csv("con_prop_cluster_df.txt", sep="\t", index=None)

print("con_prop_cluster_df")
print(con_prop_cluster_df)


# Print the properties and their corresponding cluster labels
# for property_name, cluster_label in property_cluster_map.items():
#     print(f"Property: {property_name}, Cluster: {cluster_label}")
