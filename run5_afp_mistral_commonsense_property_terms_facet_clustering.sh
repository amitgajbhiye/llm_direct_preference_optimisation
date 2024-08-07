#!/bin/bash --login

#SBATCH -A scw1858
#SBATCH --job-name=commAFP

#SBATCH --output=logs/out_run5_afp_mistral_commonsense_property_terms_facet_clustering.txt
#SBATCH --error=logs/err_run5_afp_mistral_commonsense_property_terms_facet_clustering.txt

#SBATCH --nodes=1
#SBATCH -p highmem

#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1

#SBATCH --mem=35G
#SBATCH -t 0-03:00:00

conda activate ven

# python3 src/facet_property_clustering.py --config_file configs/clustering/7_mistral7b_inst_mntp_property_terms_facet_embeds_science.json
# python3 src/facet_property_clustering.py --config_file configs/clustering/7_mistral7b_inst_mntp_property_terms_facet_embeds_food.json
# python3 src/facet_property_clustering.py --config_file configs/clustering/7_mistral7b_inst_mntp_property_terms_facet_embeds_equipment.json
# python3 src/facet_property_clustering.py --config_file configs/clustering/7_mistral7b_inst_mntp_property_terms_facet_embeds_environment.json

python3 src/facet_property_clustering.py --config_file configs/clustering/7_mistral7b_inst_mntp_property_terms_facet_embeds_commonsense.json


echo 'Job Finished !!!'
