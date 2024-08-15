#!/bin/bash --login

#SBATCH -A scw1858
#SBATCH --job-name=commAFP

#SBATCH --output=logs/out_run2_afp_mistral_food_equip_eval_5inc_facet_property_clustering.txt
#SBATCH --error=logs/err_run2_afp_mistral_food_equip_eval_5inc_facet_property_clustering.txt

#SBATCH --nodes=1
#SBATCH -p highmem

#SBATCH --ntasks=7
#SBATCH --ntasks-per-node=7
#SBATCH --cpus-per-task=1

#SBATCH --mem=150G
#SBATCH -t 0-10:00:00

conda activate venv

python3 src/facet_property_clustering.py --config_file configs/clustering/10_mistral7b_inst_mntp_facet_colon_property_embeds_5inc_food.json
python3 src/facet_property_clustering.py --config_file configs/clustering/10_mistral7b_inst_mntp_facet_colon_property_embeds_5inc_equipment.json


# python3 src/facet_property_clustering.py --config_file configs/clustering/10_mistral7b_inst_mntp_facet_colon_property_embeds_5inc_science.json
# python3 src/facet_property_clustering.py --config_file configs/clustering/10_mistral7b_inst_mntp_facet_colon_property_embeds_5inc_environment.json
# python3 src/facet_property_clustering.py --config_file configs/clustering/10_mistral7b_inst_mntp_facet_colon_property_embeds_5inc_commonsense.json

echo 'Job Finished !!!'
