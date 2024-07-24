#!/bin/bash --login

#SBATCH --job-name=afp_env

#SBATCH --output=logs/out_run1_afp_llama3_equip_science_food_taxos_clustering.txt
#SBATCH --error=logs/err_run1_afp_llama3_equip_science_food_taxos_clustering.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p highmem

#SBATCH --mem=80G
#SBATCH -t 0-10:00:00

conda activate venv

python3 src/facet_property_clustering.py --config_file configs/clustering/llama3b8_inst_mntp_embeds_equipment.json
python3 src/facet_property_clustering.py --config_file configs/clustering/llama3b8_inst_mntp_embeds_science.json
python3 src/facet_property_clustering.py --config_file configs/clustering/llama3b8_inst_mntp_embeds_food.json

echo 'Job Finished !!!'
