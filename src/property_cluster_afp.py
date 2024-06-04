import pickle

import hdbscan
import numpy as np
import pandas as pd
import torch
from llm2vec import LLM2Vec
from peft import PeftModel
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, normalize
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Loading base Mistral model, along with custom code that enables bidirectional connections in decoder-only LLMs.
tokenizer = AutoTokenizer.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
)
config = AutoConfig.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    trust_remote_code=True,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)

# Loading MNTP (Masked Next Token Prediction) model.
model = PeftModel.from_pretrained(
    model,
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
)
# Wrapper for encoding and pooling operations
l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)


# Loading transport ontology properties.
ontology_file = "data/ontology_concepts/llama3_with_3inc_exp_con_prop_parsed.txt"

df = pd.read_csv(ontology_file, sep="\t", names=["concept", "property"])

print(f"Loaded Data")
print(df)

properties = df["property"].unique()
print(f"num_properties:", properties.shape)
print(f"properties")
print(properties)

llm_prop_embeds = l2v.encode(properties).detach().cpu().numpy()

print(f"llm_prop_embeds.shape: {llm_prop_embeds.shape}")

prop_and_embedding = [(prop, embed) for prop, embed in zip(properties, llm_prop_embeds)]

print(f"Top 5 props")
print(prop_and_embedding[0:5])

pickle_output_file = "output_llm_embeds/transport_properties.pkl"
with open(pickle_output_file, "wb") as pkl_file:
    pickle.dump(prop_and_embedding, pkl_file)


# Initialize the AffinityPropagation model
affinity_propagation = AffinityPropagation()

# Fit the model to your embeddings
affinity_propagation.fit(llm_prop_embeds)

# Get the cluster centers and labels
cluster_centers_indices = affinity_propagation.cluster_centers_indices_
labels = affinity_propagation.labels_

print(f"labels")
print(labels)

property_cluster_list = [
    (prop, clus_label) for prop, clus_label in zip(properties, labels)
]

print("property_cluster_list")
print(property_cluster_list)

property_cluster_map = {
    prop: cluster_label for prop, cluster_label in zip(properties, labels)
}

print("property_cluster_map")
print(property_cluster_map)

clustered_properties = pd.DataFrame.from_records(
    property_cluster_list, columns=["property", "cluster_label"]
)

clustered_properties.sort_values(by=["cluster_label"], inplace=True)

print(f"clustered_properties")
print(clustered_properties)

clustered_properties.to_csv(
    "clustered_properties.txt",
    sep="\t",
    index=False,
)


def assign_prop_cluster_label(prop):
    return property_cluster_map[prop]


# for k, v in property_cluster_map.items():
#     print(k, v)

cluster_df = df.copy(deep=True)
prop_clusters_labels = [
    assign_prop_cluster_label(prop) for prop in cluster_df["property"].to_list()
]

cluster_df["cluster_label"] = prop_clusters_labels


cluster_df.to_csv(
    "concept_property_label.txt",
    sep="\t",
    index=False,
)

cluster_df.sort_values(by=["cluster_label"], inplace=True)

print("prop_clusters_labels")
print(prop_clusters_labels)

print("cluster_df")
print(cluster_df)
