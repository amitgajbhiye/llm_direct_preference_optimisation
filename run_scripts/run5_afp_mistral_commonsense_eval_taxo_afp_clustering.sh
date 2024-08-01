#!/bin/bash --login

#SBATCH --job-name=commAFP

#SBATCH --output=logs/out_run5_afp_mistral_commonsense_eval_taxo_afp_clustering.txt
#SBATCH --error=logs/err_run5_afp_mistral_commonsense_eval_taxo_afp_clustering.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p highmem

#SBATCH --mem=50G
#SBATCH -t 0-02:00:00

conda activate venv

python3 src/facet_property_clustering.py --config_file configs/clustering/mistral7b_inst_mntp_embeds_commonsense.json


echo 'Job Finished !!!'
