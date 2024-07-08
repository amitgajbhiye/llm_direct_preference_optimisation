import logging
import os
import pickle
import time
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler

from utilities import get_execution_time, read_config


def affinity_propagation_clustering(config):

    embedding_file = config["embedding_file"]

    with open(embedding_file, "rb") as pkl_inp:
        prop_embed = pickle.load(pkl_inp)

    properties = list(prop_embed.keys())
    prop_embeddings = np.array(list(prop_embed.values()))

    logger.info(f"embedding_file: {embedding_file}")
    logger.info(f"total_properties: {len(set(properties))}")
    logger.info(f"some_input_properties: {properties[0:10]} ...")
    logger.info(f"prop_embeddings_shape: {prop_embeddings.shape} ...")

    scaled_embeds = StandardScaler().fit_transform(prop_embeddings)

    logger.info(f"scaled_embeds.shape {scaled_embeds.shape}")

    clustering = AffinityPropagation().fit(scaled_embeds)
    labels = clustering.labels_
    total_cluster_labels = set(labels)

    logger.info(f"total_num_cluster_labels: {len(labels)}")
    logger.info(f"total_cluster_labels: {total_cluster_labels}")

    prop_cluster_list = [(prop, label) for prop, label in zip(properties, labels)]

    clusters_output_file = os.path.join(
        config["output_dir"], config["clusters_output_file"]
    )

    df = pd.DataFrame(
        prop_cluster_list, columns=["facet_property", "cluster_label"]
    ).sort_values(by=["cluster_label"])

    df.to_csv(f"{clusters_output_file}.txt", sep="\t", index=False, encoding="utf-8")

    with open(f"{clusters_output_file}.pkl", "wb") as pkl_out:
        pickle.dump(prop_cluster_list, pkl_out)

    logger.info(f"clustering_done!!!")
    logger.info(f"text_clustered_file_saved at: {clusters_output_file}.txt")
    logger.info(f"pkl_clustered_file_saved at: {clusters_output_file}.pkl")

    return f"{clusters_output_file}.txt"


def merge_concepts_clusters(all_data_file, cluster_file):

    logger.info(f"creating_final_clusters ...")

    all_data_df = pd.read_csv(all_data_file, sep="\t")

    for col_name in list(all_data_df.columns):
        all_data_df[col_name] = all_data_df[col_name].str.strip()

    logger.info(f"all_parsed_data_file: {all_data_file}")
    logger.info(f"all_parsed_data")
    logger.info(all_data_df)

    cluster_df = pd.read_csv(cluster_file, sep="\t")

    logger.info(f"cluster_file: {cluster_file}")
    logger.info(f"cluster_data")
    logger.info(cluster_df)

    cluster_labels = cluster_df["cluster_label"].unique()
    logger.info(f"cluster_labels: {len(cluster_labels), cluster_labels}")

    sorted_clusters = []
    for cluster_label in cluster_labels:

        logger.info(f"cluster_label: {cluster_label}")

        temp_df = cluster_df[cluster_df["cluster_label"] == cluster_label]
        facet_props = temp_df["facet_property"].to_list()

        # print (f"facet_prop: {facet_props}")
        for facet_prop in facet_props:

            facet_prop_list = facet_prop.split(":")
            facet_prop_list = [x.strip() for x in facet_prop_list]

            if len(facet_prop_list) > 2:
                facet = facet_prop_list[0].strip()
                prop = ":".join(facet_prop_list[1:]).strip()
            else:
                facet, prop = facet_prop_list
                facet = facet.strip()
                prop = prop.strip()

            # print (f"facet: {facet}, prop: {prop}, cluster_label: {cluster_label}")
            # print (all_data_df[(all_data_df["facet"] == facet) & (all_data_df["property"] == prop)])

            concept_clusters = all_data_df[
                (all_data_df["facet"] == facet) & (all_data_df["property"] == prop)
            ]
            concept_clusters["cluster_label"] = cluster_label
            concept_clusters["facet_property"] = facet_prop

            sorted_clusters.append(concept_clusters)

    all_clusters = pd.concat(sorted_clusters, ignore_index=True).sort_values(
        by=["cluster_label"]
    )

    all_cols_final_cluster_file = os.path.join(
        config["output_dir"], f"final_{config['clusters_output_file']}.txt"
    )
    all_clusters.to_csv(all_cols_final_cluster_file, sep="\t", index=False)

    con_prop_cluster_label_file_name = os.path.join(
        config["output_dir"],
        f"concept_property_cluster_label_{config['clusters_output_file']}.txt",
    )
    all_clusters[["concept", "property", "cluster_label"]].to_csv(
        con_prop_cluster_label_file_name, sep="\t", index=False
    )

    logger.info(f"all_cols_final_cluster_file saved at: {all_cols_final_cluster_file}")
    logger.info(
        f"con_prop_cluster_label_file_name saved at: {con_prop_cluster_label_file_name}"
    )

    return all_clusters


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

    clusters_output_file = affinity_propagation_clustering(config=config)

    merge_concepts_clusters(config["all_parsed_data_file"], clusters_output_file)

    end_time = time.time()
    get_execution_time(start_time, end_time)
