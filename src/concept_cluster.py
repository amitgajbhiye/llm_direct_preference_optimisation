import pickle

import numpy as np
import hdbscan
import pandas as pd
import torch
from llm2vec import LLM2Vec
from peft import PeftModel
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig, AutoModel, AutoTokenizer

from sklearn.metrics import silhouette_score


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


# Loading ontology concepts
ontology_file = "data/ontology_concepts/transport_vocab.txt"

with open(ontology_file, "r") as fin:
    concepts = [con.strip("\n") for con in fin.readlines()]

print(f"num_concepts: {len(concepts)}")

llm_con_embeds = l2v.encode(concepts).detach().cpu().numpy()

print(f"llm_con_embeds.shape: {llm_con_embeds.shape}")

cons_and_embedding = [(prop, embed) for prop, embed in zip(concepts, llm_con_embeds)]

print(f"Top 5 props")
print(cons_and_embedding[0:5])

pickle_output_file = "output_llm_embeds/transport_concepts.pkl"
with open(pickle_output_file, "wb") as pkl_file:
    pickle.dump(cons_and_embedding, pkl_file)


# Normalize the embeddings
# scaler = StandardScaler()
# embeddings_normalized = scaler.fit_transform(llm_con_embeds)


from sklearn.preprocessing import normalize
normalized_embeddings = normalize(llm_con_embeds, norm='l2')


cluster_algo = "HDBSCAN"

if cluster_algo == "DBSCAN":
    eps_range = [0.1, 0.3, 0.5, 0.7, 0.9]
    min_samples = [3, 5, 7, 9, 10]

    for ep in eps_range:
        for min in min_samples:

            dbscan = DBSCAN(eps=ep, min_samples=min, metric="cosine", algorithm="brute")
            clusters = dbscan.fit_predict(embeddings_normalized)

            concept_cluster_list = [
                (con, clus_label) for con, clus_label in zip(concepts, clusters)
            ]

            con_cluster_df = pd.DataFrame.from_records(
                concept_cluster_list, columns=["concept", "cluster_label"]
            )

            # property_cluster_df = pd.DataFrame.from_dict(property_cluster_map)
            con_cluster_df.sort_values(by="cluster_label", inplace=True, ascending=False)

            print(f"Eps: {ep}, Min_sam: {min}")
            print(f"con_cluster_df")
            print(con_cluster_df)

            con_cluster_df.to_csv(
                f"data/ontology_concepts/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/eps{str(ep).replace(".", "_")}_minsam{min}_transport_con_cluster_llama38b_embeds.txt",
                sep="\t",
                index=None,
            )

if cluster_algo == "HDBSCAN":

    def evaluate_hdbscan(min_cluster_size, min_samples):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='cosine')
        cluster_labels = clusterer.fit_predict(normalized_embeddings)
        
        if len(set(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(normalized_embeddings, cluster_labels)
        else:
            silhouette_avg = -1
        
        print(f"min_cluster_size: {min_cluster_size}, min_samples: {min_samples}, silhouette_score: {silhouette_avg}")
        return cluster_labels, silhouette_avg

    # Experiment with different parameter values
    results = []

    for min_cluster_size in range(2, 20):
        for min_samples in range(2, 20):
            cluster_labels, silhouette_avg = evaluate_hdbscan(min_cluster_size, min_samples)
            results.append((min_cluster_size, min_samples, silhouette_avg, cluster_labels))

    # Choose the best parameters based on silhouette score
    best_result = max(results, key=lambda x: x[2])
    best_min_cluster_size, best_min_samples, best_silhouette, best_cluster_labels = best_result

    print(f"Best min_cluster_size: {best_min_cluster_size}, Best min_samples: {best_min_samples}, Best silhouette_score: {best_silhouette}")


















    # min_cluster_size = []
    # min_samples = []
    
    # clusters = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
    # cluster_labels = clusters.fit_predict(llm_con_embeds)

    # concept_cluster_list = [
    # (con, clus_label) for con, clus_label in zip(concepts, cluster_labels)
    # ]

    # con_cluster_df = pd.DataFrame.from_records(
    #     concept_cluster_list, columns=["concept", "cluster_label"]
    # )

    # con_cluster_df.sort_values(by="cluster_label", inplace=True, ascending=False)

    # print(f"con_cluster_df")
    # print(con_cluster_df)

    # con_cluster_df.to_csv(
    #     f"data/ontology_concepts/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/hdbscan_transport_con_cluster_llama38b_embeds.txt",
    #     sep="\t",
    #     index=None,
    # )




