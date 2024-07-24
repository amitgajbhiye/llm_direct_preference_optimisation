#!/bin/bash --login

#SBATCH --job-name=afp_env

#SBATCH --output=logs/out_llama3b8_inst_mntp_embeds_environment.txt
#SBATCH --error=logs/err_llama3b8_inst_mntp_embeds_environment.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p highmem

#SBATCH --mem=75G
#SBATCH -t 0-04:00:00

conda activate venv

python3 src/facet_property_clustering.py --config_file configs/clustering/llama3b8_inst_mntp_embeds_environment.json

echo 'Job Finished !!!'
