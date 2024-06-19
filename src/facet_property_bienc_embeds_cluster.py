import pickle

import hdbscan
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# file = "/home/amit/cardiff_work/llm_direct_preference_optimisation/output_llm_embeds/bienc_embeds/bienc_entropy_bert_large_cnetpchatgpt_llama3_facet_property_property_embeddings.pkl"

file = "/home/amit/cardiff_work/llm_direct_preference_optimisation/output_llm_embeds/bienc_embeds/bienc_entropy_bert_large_cnetpchatgpt_llama3_facet_colon_sep_property_property_embeddings.pkl"
hawk_file = "/scratch/c.scmag3/property_augmentation/trained_models/embeds_for_commonalities/bienc_entropy_bert_large_cnetpchatgpt_llama3_facet_colon_sep_property_property_embeddings.pkl"

wine_hawk_file = "/scratch/c.scmag3/property_augmentation/trained_models/embeds_for_commonalities/bienc_entropy_bert_large_cnetpchatgpt_llama3_wine_facet_colon__property_embeddings.pkl"

with open(wine_hawk_file, "rb") as pkl_inp:
    prop_embed = pickle.load(pkl_inp)

properties = list(prop_embed.keys())
prop_embeddings = np.array(list(prop_embed.values()))

print(f"Properties: {properties[0:10]} ...")
print(f"prop_embeddings: {prop_embeddings.shape} ...")
print(f"Properties: {len(set(properties))}")


cluster_algo = "affinity_propogation"
print(f"cluster_algo:", cluster_algo)

if cluster_algo == "affinity_propogation":

    X = StandardScaler().fit_transform(prop_embeddings)

    print(f"X.shape {X.shape}")

    clustering = AffinityPropagation().fit(X)
    labels = clustering.labels_

    print(f"labels: {len(set(labels))}")
    print(f"labels: {set(labels)}")

    prop_cluster_list = [(prop, label) for prop, label in zip(properties, labels)]

    for prop, cluster in prop_cluster_list:
        print(prop, cluster)

    output_file = "bienc_entropy_bert_large_cnetpchatgpt_llama3_wine_facet_colon_property_embeddings_affinity_propogation_clusters"

    df = pd.DataFrame(
        prop_cluster_list, columns=["facet_property", "cluster_label"]
    ).sort_values(by=["cluster_label"])

    df.to_csv(
        f"{output_file}.txt",
        sep="\t",
        index=False,
    )

    with open(f"{output_file}.pkl", "wb") as pkl_out:
        pickle.dump(prop_cluster_list, pkl_out)


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

    # eps_values = np.arange(0.1, 1.1, 0.1)
    # min_samples_values = range(2, 10)

    eps_values = [0.5]
    min_samples_values = [5]

    X = StandardScaler().fit_transform(prop_embeddings)

    best_score, best_params, best_labels = dbscan_clustering(
        X, eps_values, min_samples_values
    )

    print(f"Best Silhouette Score: {best_score}")
    print(f"Best Parameters: eps={best_params[0]}, min_samples={best_params[1]}")

    # Best Silhouette Score: 0.02702467143535614
    # Best Parameters: eps=0.5, min_samples=5

    with open("bienc_cluster_colon_sep.txt", "w") as out_file:
        for prop, label in zip(properties, best_labels):
            out_file.write(f"{prop}\t{label}\n")


if cluster_algo == "HDBSCAN":

    def hdbscan_clustering(X, min_cluster_size_values, min_samples_values):
        print(f"In hdbscan_clustering")
        best_score = -1
        best_params = None
        best_labels = None

        for min_cluster_size in min_cluster_size_values:
            for min_samples in min_samples_values:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size, min_samples=min_samples
                )
                labels = clusterer.fit_predict(X)

                # Ignore clusters where all points are classified as noise
                if len(set(labels)) <= 1 or (len(set(labels)) == 2 and -1 in labels):
                    continue

                # Calculate silhouette score
                score = silhouette_score(X, labels)

                # Check if we got a better score
                if score > best_score:
                    best_score = score
                    best_params = (min_cluster_size, min_samples)
                    best_labels = labels

                print(f"min_cluster_size: {min_cluster_size}")
                print(f"min_samples: {min_samples}")
                print(f"score: {score}")
                print(f"best_score: {best_score}")
                print()

        return best_score, best_params, best_labels

    min_cluster_size_values = range(3, 10)
    min_samples_values = range(1, 10)

    X = StandardScaler().fit_transform(prop_embeddings)

    best_score, best_params, best_labels = hdbscan_clustering(
        X, min_cluster_size_values, min_samples_values
    )

    print(f"Best Silhouette Score: {best_score}")
    print(
        f"Best Parameters: min_cluster_size={best_params[0]}, min_samples={best_params[1]}"
    )
