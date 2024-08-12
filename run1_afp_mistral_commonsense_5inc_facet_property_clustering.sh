#!/bin/bash --login

#SBATCH -A scw1858
#SBATCH --job-name=commAFP

#SBATCH --output=logs/out_run1_afp_mistral_commonsense_5inc_facet_property_clustering.txt
#SBATCH --error=logs/err_run1_afp_mistral_commonsense_5inc_facet_property_clustering.txt

#SBATCH --nodes=1
#SBATCH -p highmem

#SBATCH --ntasks=7
#SBATCH --ntasks-per-node=7
#SBATCH --cpus-per-task=1

#SBATCH --mem=50G
#SBATCH -t 0-02:00:00

conda activate venv

python3 src/facet_property_clustering.py --config_file configs/clustering/10_mistral7b_inst_mntp_facet_colon_property_embeds_5inc_commonsense.json

echo 'Job Finished !!!'
