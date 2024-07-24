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

    if isinstance(prop_embed, list):
        logger.info(f"converting (prop, embed) lost to prop: embed dict")
        prop_embed = {facet_prop: embed for facet_prop, embed in prop_embed}

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
        f"final_concept_property_cluster_label_{config['clusters_output_file']}.txt",
    )
    all_clusters[["concept", "property", "cluster_label"]].to_csv(
        con_prop_cluster_label_file_name, sep="\t", index=False
    )

    logger.info(f"all_cols_final_cluster_file saved at: {all_cols_final_cluster_file}")
    logger.info(
        f"con_prop_cluster_label_file_name saved at: {con_prop_cluster_label_file_name}"
    )

    return all_cols_final_cluster_file, con_prop_cluster_label_file_name


def max_jaccard_gold_and_predicted_clusters(
    con_prop_cluster_label_file, taxo_file, output_file
):
    # Files required for
    # cluster_file = "data/evaluation_taxo/generated_facet_property/final_concept_property_cluster_label_bienc_commonsense_facet_property_embeds_afp_clusters.txt"
    # taxo_file = "data/evaluation_taxo/gold/commonsense.taxo"
    # final_output_file = "commonsense_taxo_commonalities_max_jaccard_gold_and_predicted.txt"

    cluster_df = pd.read_csv(con_prop_cluster_label_file, sep="\t")
    taxo_df = pd.read_csv(taxo_file, sep="\t", names=["concept", "property"])

    print("taxo_df")
    print(taxo_df)
    print(len(taxo_df["concept"].unique()), taxo_df["concept"].unique())

    taxo_prop = taxo_df["property"].unique().tolist()
    predicted_cluster_labels = cluster_df["cluster_label"].unique()

    jaccard_indices = []

    print(f"taxo_prop: {len(taxo_prop)}, {taxo_prop}")
    print(
        f"predicted_cluster_labels: {len(predicted_cluster_labels)}, {predicted_cluster_labels}"
    )

    with open(output_file, "w") as out_file:
        for prop in taxo_prop:
            prop_results = []

            taxo_con = set(taxo_df[taxo_df["property"] == prop]["concept"].unique())
            for cluster_label in predicted_cluster_labels:
                predicted_cluster_con = set(
                    cluster_df[cluster_df["cluster_label"] == int(cluster_label)][
                        "concept"
                    ].unique()
                )

                intersection = predicted_cluster_con.intersection(taxo_con)
                union = predicted_cluster_con.union(taxo_con)

                jaccard_score = round(len(intersection) / len(union), 4)

                prop_results.append((cluster_label, jaccard_score))

            max_jaccard_index = max(prop_results, key=lambda x: x[1])[1]
            max_jaccard_tuples = [t for t in prop_results if t[1] == max_jaccard_index]
            jaccard_indices.append(max_jaccard_index)

            if max_jaccard_index != 0.0:
                max_jaccard_cluster_labels = [t[0] for t in max_jaccard_tuples]

                print(f"{'*'*20}")
                print(f"**gold_taxo_property: {prop}")
                print(f"**gold_taxo_cons: {sorted(taxo_con)}")
                print(f"**best_jaccard_index: {max_jaccard_index}")
                print(f"**best_jaccard_clusters: {max_jaccard_cluster_labels}")

                out_file.write(f"{'*'*20}\n")
                out_file.write(f"**gold_taxo_property: {prop}\n")
                out_file.write(f"**gold_taxo_cons: {sorted(taxo_con)}\n")
                out_file.write(f"**best_jaccard_index: {max_jaccard_index}\n")
                out_file.write(
                    f"**best_jaccard_clusters: {max_jaccard_cluster_labels}\n"
                )

                for cluster_label in max_jaccard_cluster_labels:

                    predicted_cluster_con = sorted(
                        cluster_df[cluster_df["cluster_label"] == int(cluster_label)][
                            "concept"
                        ].unique()
                    )
                    predicted_cluster_prop = cluster_df[
                        cluster_df["cluster_label"] == int(cluster_label)
                    ]["property"].unique()

                    extra_cons_predicted_cluster = [
                        item for item in predicted_cluster_con if item not in taxo_con
                    ]
                    common_cons_predicted_gold_cons = [
                        item for item in predicted_cluster_con if item in taxo_con
                    ]

                    print(f"**cluster_label: {cluster_label}")
                    print(f"**predicted_cluster_con: {sorted(predicted_cluster_con)}")
                    print(f"**predicted_cluster_prop: {predicted_cluster_prop}")
                    print(f"**con_intersection: {common_cons_predicted_gold_cons}")
                    print(
                        f"**extra_cons_in_predicted_cluster: {extra_cons_predicted_cluster}"
                    )
                    print()

                    out_file.write(f"**cluster_label: {cluster_label}\n")
                    out_file.write(
                        f"**predicted_cluster_con: {sorted(predicted_cluster_con)}\n"
                    )
                    out_file.write(
                        f"**predicted_cluster_prop: {predicted_cluster_prop}\n"
                    )
                    out_file.write(
                        f"**con_intersection: {common_cons_predicted_gold_cons}\n"
                    )
                    out_file.write(
                        f"**extra_cons_in_predicted_cluster: {extra_cons_predicted_cluster}\n"
                    )
                    out_file.write(f"\n")

            else:

                print(f"{'*'*20}")
                print(f"**gold_taxo_property: {prop}")
                print(f"**gold_taxo_cons: {sorted(taxo_con)}")
                print(f"**best_jaccard_index: {max_jaccard_index}")
                print()

                out_file.write(f"{'*'*20}\n")
                out_file.write(f"**gold_taxo_property: {prop}\n")
                out_file.write(f"**gold_taxo_cons: {sorted(taxo_con)}\n")
                out_file.write(f"**best_jaccard_index: {max_jaccard_index}\n")
                out_file.write(f"\n")

        avg_jaccard_index = round(np.average(jaccard_indices), 4)
        out_file.write(f"**avg_jaccard_index: {avg_jaccard_index}\n")

        print(f"**avg_jaccard_index: {avg_jaccard_index}")

        logger.info(f"max_jaccard_gold_and_predicted_clusters saved at: {output_file}")


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

    all_cols_final_cluster_file, con_prop_cluster_label_file_name = (
        merge_concepts_clusters(config["all_parsed_data_file"], clusters_output_file)
    )

    logger.info(f"calculating max_jaccard_gold_and_predicted_clusters")

    taxo_file = config["taxo_file"]
    output_file = os.path.join(
        config["output_dir"], config["max_jaccard_gold_and_predicted_clusters_file"]
    )

    max_jaccard_gold_and_predicted_clusters(
        con_prop_cluster_label_file=con_prop_cluster_label_file_name,
        taxo_file=taxo_file,
        output_file=output_file,
    )

    end_time = time.time()
    get_execution_time(start_time, end_time)
