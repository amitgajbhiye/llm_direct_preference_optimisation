#!/bin/bash --login

#SBATCH -A scw1858
#SBATCH --job-name=commAFP

#SBATCH --output=logs/out_run7_afp_mistral_all_eval_property_feature_facet_clustering.txt
#SBATCH --error=logs/err_run7_afp_mistral_all_eval_property_feature_facet_clustering.txt

#SBATCH --nodes=1
#SBATCH -p highmem

#SBATCH --ntasks=10
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1

#SBATCH --mem=70G
#SBATCH -t 0-12:00:00

conda activate venv


python3 src/facet_property_clustering.py --config_file configs/clustering/8_mistral7b_inst_mntp_property_feature_facet_embeds_food.json
python3 src/facet_property_clustering.py --config_file configs/clustering/8_mistral7b_inst_mntp_property_feature_facet_embeds_equipment.json
python3 src/facet_property_clustering.py --config_file configs/clustering/8_mistral7b_inst_mntp_property_feature_facet_embeds_science.json
python3 src/facet_property_clustering.py --config_file configs/clustering/8_mistral7b_inst_mntp_property_feature_facet_embeds_environment.json
python3 src/facet_property_clustering.py --config_file configs/clustering/8_mistral7b_inst_mntp_property_feature_facet_embeds_commonsense.json

echo 'Job Finished !!!'
