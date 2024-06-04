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


# cluster_algo = "HDBSCAN"

# if cluster_algo == "DBSCAN":

#     # Normalize the embeddings
#     scaler = StandardScaler()
#     embeddings_normalized = scaler.fit_transform(llm_con_embeds)


#     eps_range = [0.1, 0.3, 0.5, 0.7, 0.9]
#     min_samples = [3, 5, 7, 9, 10]

#     for ep in eps_range:
#         for min in min_samples:

#             dbscan = DBSCAN(eps=ep, min_samples=min, metric="cosine", algorithm="brute")
#             clusters = dbscan.fit_predict(embeddings_normalized)

#             concept_cluster_list = [
#                 (con, clus_label) for con, clus_label in zip(concepts, clusters)
#             ]

#             con_cluster_df = pd.DataFrame.from_records(
#                 concept_cluster_list, columns=["concept", "cluster_label"]
#             )

#             # property_cluster_df = pd.DataFrame.from_dict(property_cluster_map)
#             con_cluster_df.sort_values(by="cluster_label", inplace=True, ascending=False)

#             print(f"Eps: {ep}, Min_sam: {min}")
#             print(f"con_cluster_df")
#             print(con_cluster_df)

#             con_cluster_df.to_csv(
#                 f"data/ontology_concepts/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/eps{str(ep).replace(".", "_")}_minsam{min}_transport_con_cluster_llama38b_embeds.txt",
#                 sep="\t",
#                 index=None,
#             )

# if cluster_algo == "HDBSCAN":

#     normalized_embeddings = normalize(llm_con_embeds, norm='l2')

#     # normalized_embeddings = llm_con_embeds

#     # cosine_distances = pdist(normalized_embeddings, metric='cosine')
#     # cosine_distance_matrix = squareform(cosine_distances)

#     def evaluate_hdbscan(min_cluster_size, min_samples):

#         clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
#         cluster_labels = clusterer.fit_predict(normalized_embeddings)

#         if len(set(cluster_labels)) > 1:
#             silhouette_avg = silhouette_score(normalized_embeddings, cluster_labels)
#         else:
#             silhouette_avg = -1

#         print(f"min_cluster_size: {min_cluster_size}, min_samples: {min_samples}, silhouette_score: {silhouette_avg}")
#         return cluster_labels, silhouette_avg

#     # Experiment with different parameter values
#     # results = []

#     # for min_cluster_size in range(2, 20):
#     #     for min_samples in range(2, 20):
#     #         cluster_labels, silhouette_avg = evaluate_hdbscan(min_cluster_size, min_samples)
#     #         results.append((min_cluster_size, min_samples, silhouette_avg, cluster_labels))


#     cluster_labels, silhouette_avg = evaluate_hdbscan(min_cluster_size=3, min_samples=2)

#     # Choose the best parameters based on silhouette score
#     # best_result = max(results, key=lambda x: x[2])
#     # best_min_cluster_size, best_min_samples, best_silhouette, best_cluster_labels = best_result


#     # print(f"Best min_cluster_size: {best_min_cluster_size}, Best min_samples: {best_min_samples}, Best silhouette_score: {best_silhouette}")
#     print(f"Best silhouette_score: {silhouette_avg}")

#     concept_cluster_list = [
#     (con, clus_label) for con, clus_label in zip(concepts, cluster_labels)
# ]

#     con_cluster_df = pd.DataFrame.from_records(
#         concept_cluster_list, columns=["concept", "cluster_label"]
#     )

#     # property_cluster_df = pd.DataFrame.from_dict(property_cluster_map)
#     con_cluster_df.sort_values(by="cluster_label", inplace=True, ascending=False)

#     print(con_cluster_df)

#     con_cluster_df.to_csv(
#         f"data/ontology_concepts/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/hbdscan_transport_con_cluster_llama38b_embeds.txt",
#         sep="\t",
#         index=None,
#     )
