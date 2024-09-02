#!/bin/bash --login

#SBATCH -A scw1858
#SBATCH --job-name=ufetAFP

#SBATCH --output=logs/ufet_clustering/out_run7_afp_mistral_ufet_1inc_facet_property_clustering.txt
#SBATCH --error=logs/ufet_clustering/err_run7_afp_mistral_ufet_1inc_facet_property_clustering.txt

#SBATCH --nodes=1
#SBATCH -p highmem

#SBATCH --ntasks=7
#SBATCH --ntasks-per-node=7
#SBATCH --cpus-per-task=1

#SBATCH --mem=299G
#SBATCH -t 0-21:00:00

conda activate venv

python3 src/facet_property_clustering.py --config_file configs/clustering/12_ufet_mistral7b_inst_mntp_facet_colon_property_embeds_1inc_repeat5.json

echo 'Job Finished !!!'
