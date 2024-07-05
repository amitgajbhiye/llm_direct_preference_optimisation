import pickle
from collections import Counter

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
concept_facet_property_file = "data/ontology_concepts/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/facet_property/llama3_concept_facet_property_transport_onto_concepts_parsed.txt"

df = pd.read_csv(concept_facet_property_file, sep="\t")

# print(f"loaded: concept_facet_property_file")
# print(df)

uniq_concepts = df["concept"].unique()
# print(f"uniq_num_concepts:", uniq_concepts.shape)

uniq_facets = df["facet"].unique()
# print(f"num_facets:", uniq_facets.shape)

facet_count_dict = dict((Counter(df["facet"].to_list())))

uniq_properties = df["property"].unique()
# print(f"num_properties:", uniq_properties.shape)
# print()
# print()

cluster_all_properties = True
if cluster_all_properties:

    # print(f"num_property_for_facet: {len(properties)}")

    llm_prop_embeds = l2v.encode(uniq_properties).detach().cpu().numpy()

    # print(f"llm_prop_embeds.shape: {llm_prop_embeds.shape}")

    prop_and_embedding = [
        (prop, embed) for prop, embed in zip(uniq_properties, llm_prop_embeds)
    ]

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(llm_prop_embeds)

    clustering_algorithm = "dbscan"

    if clustering_algorithm == "affinity_propagation":
        affinity_propagation = AffinityPropagation()
        # Fit the model to your embeddings
        affinity_propagation.fit(embeddings_scaled)

        # Get the cluster centers and labels
        cluster_centers_indices = affinity_propagation.cluster_centers_indices_
        labels = affinity_propagation.labels_

    elif clustering_algorithm == "dbscan":

        dbscan_clusterer = DBSCAN(
            eps=0.5, min_samples=3, metric="cosine", algorithm="brute"
        )
        # dbscan_clusterer = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan_clusterer.fit_predict(embeddings_scaled)

    elif clustering_algorithm == "hdbscan":

        # hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
        hdbscan_clusterer = hdbscan.HDBSCAN()
        labels = hdbscan_clusterer.fit_predict(embeddings_scaled)

    property_cluster_list = [
        (prop, clus_label) for prop, clus_label in zip(uniq_properties, labels)
    ]

    # sorted_property_cluster_list = sorted(
    #     property_cluster_list, key=lambda x: x[1], reverse=True
    # )

    # print("property", "\t", "cluster_label")
    # for prop, cluster in sorted_property_cluster_list:
    #     print(prop, "\t", cluster)

    property_cluster_map = {
        prop: clus_label for prop, clus_label in zip(uniq_properties, labels)
    }

    for i, facet in enumerate(uniq_facets):
        facet_count = facet_count_dict[facet]

        print(
            f"****** Processing facet_no: {i}, facet: {facet}, facet_count: {facet_count} ******"
        )

        if facet_count < 2:
            print(f"facet_count: {facet_count}, less than 2; ignoring facet")
            continue

        facet_properties = df[df["facet"] == facet]["property"].unique()

        property_cluster_list = [
            (facet_prop, property_cluster_map[facet_prop])
            for facet_prop in facet_properties
        ]

        sorted_property_cluster_list = sorted(
            property_cluster_list, key=lambda x: x[1], reverse=True
        )

        print("property", "\t", "cluster_label")
        for prop, cluster_label in sorted_property_cluster_list:
            print(prop, "\t", cluster_label)

        print()
        print()


######
# for i, facet in enumerate(uniq_facets):

#     facet_count = facet_count_dict[facet]

#     print(
#         f"****** Processing facet_no: {i}, facet: {facet}, facet_count: {facet_count} ******"
#     )

#     if facet_count < 2:
#         print(f"facet_count: {facet_count}, less than 2; ignoring facet")
#         continue

#     # print(f"facet_count: {facet_count}")
#     properties = df[df["facet"] == facet]["property"].unique()

#     # print(f"num_property_for_facet: {len(properties)}")

#     llm_prop_embeds = l2v.encode(properties).detach().cpu().numpy()

#     # print(f"llm_prop_embeds.shape: {llm_prop_embeds.shape}")

#     prop_and_embedding = [
#         (prop, embed) for prop, embed in zip(properties, llm_prop_embeds)
#     ]

#     scaler = StandardScaler()
#     embeddings_scaled = scaler.fit_transform(llm_prop_embeds)

#     clustering_algorithm = "hdbscan"

#     if clustering_algorithm == "affinity_propagation":
#         affinity_propagation = AffinityPropagation()
#         # Fit the model to your embeddings
#         affinity_propagation.fit(embeddings_scaled)

#         # Get the cluster centers and labels
#         cluster_centers_indices = affinity_propagation.cluster_centers_indices_
#         labels = affinity_propagation.labels_

#     elif clustering_algorithm == "dbscan":

#         # dbscan_clusterer = DBSCAN(
#         #     eps=0.5, min_samples=3, metric="cosine", algorithm="brute"
#         # )
#         dbscan_clusterer = DBSCAN(eps=0.5, min_samples=5)
#         labels = dbscan_clusterer.fit_predict(embeddings_scaled)

#     elif clustering_algorithm == "hdbscan":

#         # hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
#         hdbscan_clusterer = hdbscan.HDBSCAN()
#         labels = hdbscan_clusterer.fit_predict(embeddings_scaled)

#     property_cluster_list = [
#         (prop, clus_label) for prop, clus_label in zip(properties, labels)
#     ]

#     sorted_property_cluster_list = sorted(property_cluster_list, key=lambda x: x[1])

#     print("property", "\t", "cluster_label")
#     for prop, cluster in sorted_property_cluster_list:
#         print(prop, "\t", cluster)

#     # print("sorted_property_cluster_list")
#     # print(sorted_property_cluster_list)

#     print(f"****** Finished processing facet: {facet} ******")
#     print()
#     print()