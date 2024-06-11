import pickle

import hdbscan
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

file = "../output_llm_embeds/bienc_embeds/bienc_entropy_bert_large_cnetpchatgpt_llama3_facet_property_property_embeddings.pkl"

with open(file, "rb") as pkl_inp:
    prop_embed = pickle.load(pkl_inp)

properties = list(prop_embed.keys())
prop_embeddings = np.array(list(prop_embed.values()))

print(f"Properties: {properties[0:10]} ...")
print(f"prop_embeddings: {prop_embeddings.shape} ...")

cluster_algo = "DBSCAN"

if cluster_algo == "DBSCAN":

    def dbscan_clustering(X, eps_values, min_samples_values):

        best_score = -1
        best_params = None
        best_labels = None

        for eps in eps_values:
            for min_samples in min_samples_values:
                db = DBSCAN(
                    eps=eps, min_samples=min_samples, metric="cosine", algorithm="brute"
                ).fit(X)
                labels = db.labels_

                # Ignore clusters where all points are classified as noise
                if len(set(labels)) <= 1:
                    continue

                # Calculate silhouette score
                score = silhouette_score(X, labels)

                # Check if we got a better score
                if score > best_score:
                    best_score = score
                    best_params = (eps, min_samples)
                    best_labels = labels

        return best_score, best_params, best_labels

    eps_values = np.arange(0.1, 1.5, 0.1)
    min_samples_values = range(2, 10)

    X = StandardScaler().fit_transform(prop_embeddings)

    best_score, best_params, best_labels = dbscan_clustering(
        X, eps_values, min_samples_values
    )

    print(f"Best Silhouette Score: {best_score}")
    print(f"Best Parameters: eps={best_params[0]}, min_samples={best_params[1]}")

    for prop, label in zip(properties, best_labels):
        print(prop, "\t", label)


# if cluster_algo == "HDBSCAN":

#     normalized_embeddings = normalize(llm_con_embeds, norm="l2")

#     # normalized_embeddings = llm_con_embeds

#     # cosine_distances = pdist(normalized_embeddings, metric='cosine')
#     # cosine_distance_matrix = squareform(cosine_distances)

#     def evaluate_hdbscan(min_cluster_size, min_samples):

#         clusterer = hdbscan.HDBSCAN(
#             min_cluster_size=min_cluster_size, min_samples=min_samples
#         )
#         cluster_labels = clusterer.fit_predict(normalized_embeddings)

#         if len(set(cluster_labels)) > 1:
#             silhouette_avg = silhouette_score(normalized_embeddings, cluster_labels)
#         else:
#             silhouette_avg = -1

#         print(
#             f"min_cluster_size: {min_cluster_size}, min_samples: {min_samples}, silhouette_score: {silhouette_avg}"
#         )
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
#         (con, clus_label) for con, clus_label in zip(concepts, cluster_labels)
#     ]

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
