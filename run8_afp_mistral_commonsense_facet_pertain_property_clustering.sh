#!/bin/bash --login

#SBATCH -A scw1858
#SBATCH --job-name=commAFP

#SBATCH --output=logs/out_run8_afp_mistral_commonsense_facet_pertain_property_clustering.txt
#SBATCH --error=logs/err_run8_afp_mistral_commonsense_facet_pertain_property_clustering.txt

#SBATCH --nodes=1
#SBATCH -p dev

#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=1

#SBATCH --mem=35G
#SBATCH -t 0-01:00:00

conda activate venv

python3 src/facet_property_clustering.py --config_file configs/clustering/9_mistral7b_inst_mntp_facet_pertain_property_embeds_commonsense.json

echo 'Job Finished !!!'
