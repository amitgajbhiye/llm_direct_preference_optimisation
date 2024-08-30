import time
import warnings

import numpy as np
import pandas as pd

from utilities import get_execution_time

warnings.filterwarnings("ignore")


def max_jaccard_gold_and_predicted_clusters(con_prop_gen_file, taxo_file, output_file):

    # Files required for
    # cluster_file = "data/evaluation_taxo/generated_facet_property/final_concept_property_cluster_label_bienc_commonsense_facet_property_embeds_afp_clusters.txt"
    # taxo_file = "data/evaluation_taxo/gold/commonsense.taxo"
    # final_output_file = "commonsense_taxo_commonalities_max_jaccard_gold_and_predicted.txt"

    print(f"calculating max_jaccard_gold_and_predicted_clusters")

    gen_df = pd.read_csv(con_prop_gen_file, sep="\t")
    taxo_df = pd.read_csv(taxo_file, sep="\t", names=["concept", "property"])

    print("taxo_df")
    print(taxo_df)
    print(len(taxo_df["concept"].unique()), taxo_df["concept"].unique())

    taxo_prop = taxo_df["property"].unique().tolist()

    jaccard_indices = []

    print(f"taxo_prop: {len(taxo_prop)}, {taxo_prop}")

    with open(output_file, "w") as out_file:
        for prop in taxo_prop:
            prop_results = []

            taxo_con = set(taxo_df[taxo_df["property"] == prop]["concept"].unique())
            gen_con = set(gen_df[gen_df["property"] == prop]["concept"].unique())

            intersection = gen_con.intersection(taxo_con)
            union = gen_con.union(taxo_con)

            jaccard_score = round(len(intersection) / len(union), 4)
            jaccard_indices.append(jaccard_score)

            if jaccard_score != 0.0:

                print(f"{'*'*20}")
                print(f"**gold_taxo_property: {prop}")
                print(f"**gold_taxo_cons: {sorted(list(taxo_con))}")
                print(f"**jaccard_index: {jaccard_score}")
                print(f"**predicted_cluster_con: {sorted(list(gen_con))}")
                print(f"**con_intersection: {intersection}")
                print()

                out_file.write(f"{'*'*20}\n")
                out_file.write(f"**gold_taxo_property: {prop}\n")
                out_file.write(f"**gold_taxo_cons: {sorted(list(taxo_con))}\n")
                out_file.write(f"**jaccard_index: {jaccard_score}\n")
                out_file.write(f"**predicted_cluster_con: {sorted(list(gen_con))}\n")
                out_file.write(f"**con_intersection: {intersection}")
                out_file.write(f"\n")

            else:

                print(f"{'*'*20}")
                print(f"**gold_taxo_property: {prop}")
                print(f"**gold_taxo_cons: {sorted(list(taxo_con))}")
                print(f"**jaccard_index: {jaccard_score}")
                print(f"**predicted_cluster_con: {sorted(list(gen_con))}")
                print(f"**con_intersection: {intersection}")
                print()

                out_file.write(f"{'*'*20}\n")
                out_file.write(f"**gold_taxo_property: {prop}\n")
                out_file.write(f"**gold_taxo_cons: {sorted(list(taxo_con))}\n")
                out_file.write(f"**jaccard_index: {jaccard_score}\n")
                out_file.write(f"**predicted_cluster_con: {sorted(list(gen_con))}\n")
                out_file.write(f"**con_intersection: {intersection}")
                out_file.write(f"\n")

        avg_jaccard_index = round(np.average(jaccard_indices), 4)
        out_file.write(f"**avg_jaccard_index: {avg_jaccard_index}\n")

        print(f"**avg_jaccard_index: {avg_jaccard_index}")

        print(f"calculating max_jaccard_gold_and_predicted_clusters - Done !!")
        print(f"max_jaccard_gold_and_predicted_clusters saved at: {output_file}")


if __name__ == "__main__":
    start_time = time.time()

    # con_prop_cluster_label_file_name = "data/evaluation_taxo/generated_facet_property/lama3_concept_facet_property_commonsense_taxo_concepts_parsed.txt"
    # taxo_file = "data/evaluation_taxo/gold/commonsense.taxo"
    # output_file = "max_jaccard_commonsense_baseline_results.txt"

    # con_prop_cluster_label_file_name = "data/evaluation_taxo/generated_facet_property/lama3_concept_facet_property_food_taxo_concepts_parsed.txt"
    # taxo_file = "data/evaluation_taxo/gold/food.taxo"
    # output_file = "max_jaccard_food_baseline_results.txt"

    # con_prop_cluster_label_file_name = "data/evaluation_taxo/generated_facet_property/lama3_concept_facet_property_equipment_taxo_concepts_parsed.txt"
    # taxo_file = "data/evaluation_taxo/gold/equipment.taxo"
    # output_file = "max_jaccard_equipment_baseline_results.txt"

    # con_prop_cluster_label_file_name = "data/evaluation_taxo/generated_facet_property/lama3_concept_facet_property_science_taxo_concepts_parsed.txt"
    # taxo_file = "data/evaluation_taxo/gold/science.taxo"
    # output_file = "max_jaccard_science_baseline_results.txt"

    con_prop_cluster_label_file_name = "data/evaluation_taxo/generated_facet_property/lama3_concept_facet_property_environment_taxo_concepts_parsed.txt"
    taxo_file = "data/evaluation_taxo/gold/environment.taxo"
    output_file = "max_jaccard_environment_baseline_results.txt"

    max_jaccard_gold_and_predicted_clusters(
        con_prop_gen_file=con_prop_cluster_label_file_name,
        taxo_file=taxo_file,
        output_file=output_file,
    )

    end_time = time.time()
    get_execution_time(start_time, end_time)
