#!/bin/bash --login

#SBATCH -A scw1858
#SBATCH --job-name=commAFP

#SBATCH --output=logs/out_run2_afp_mistral_property_embeds_commonsense_clustering.txt
#SBATCH --error=logs/err_run2_afp_mistral_property_embeds_commonsense_clustering.txt

#SBATCH --nodes=1
#SBATCH -p highmem

#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1


#SBATCH --mem=30G
#SBATCH -t 0-04:00:00

conda activate venv

python3 src/facet_property_clustering.py --config_file configs/clustering/6_mistral7b_inst_mntp_property_embeds_commonsense.json

echo 'Job Finished !!!'
