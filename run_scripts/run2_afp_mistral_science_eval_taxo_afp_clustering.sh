#!/bin/bash --login

#SBATCH --job-name=scAFP

#SBATCH --output=logs/out_run2_afp_mistral_science_eval_taxo_afp_clustering.txt
#SBATCH --error=logs/err_run2_afp_mistral_science_eval_taxo_afp_clustering.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

##SBATCH -p gpu_v100
##SBATCH --mem=35G
##SBATCH --gres=gpu:1
#SBATCH -t 0-03:00:00

conda activate venv

python3 src/facet_property_clustering.py --config_file configs/clustering/mistral7b_inst_mntp_embeds_science.json


echo 'Job Finished !!!'
