#!/bin/bash --login

#SBATCH -A scw1858
#SBATCH --job-name=commAFP

#SBATCH --output=logs/out_run9_afp_mistral_all_eval_facet_pertain_property_clustering.txt
#SBATCH --error=logs/err_run9_afp_mistral_all_eval_facet_pertain_property_clusteringtxt

#SBATCH --nodes=1
#SBATCH -p highmem

#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=1

#SBATCH --mem=70G
#SBATCH -t 0-13:00:00

conda activate venv

python3 src/facet_property_clustering.py --config_file configs/clustering/9_mistral7b_inst_mntp_facet_pertain_property_embeds_environment.json
python3 src/facet_property_clustering.py --config_file configs/clustering/9_mistral7b_inst_mntp_facet_pertain_property_embeds_equipment.json
python3 src/facet_property_clustering.py --config_file configs/clustering/9_mistral7b_inst_mntp_facet_pertain_property_embeds_food.json
python3 src/facet_property_clustering.py --config_file configs/clustering/9_mistral7b_inst_mntp_facet_pertain_property_embeds_science.json

echo 'Job Finished !!!'
