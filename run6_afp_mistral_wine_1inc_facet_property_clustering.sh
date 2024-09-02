#!/bin/bash --login

#SBATCH -A scw1858
#SBATCH --job-name=wineAFP

#SBATCH --output=logs/ontology_clustering/out_run6_afp_mistral_wine_1inc_facet_property_clustering.sh
#SBATCH --error=logs/ontology_clustering/err_run6_afp_mistral_wine_1inc_facet_property_clustering.sh

#SBATCH --nodes=1
#SBATCH -p highmem

#SBATCH --ntasks=7
#SBATCH --ntasks-per-node=7
#SBATCH --cpus-per-task=1

#SBATCH --mem=75G
#SBATCH -t 0-08:00:00

conda activate venv

python3 src/facet_property_clustering.py --config_file configs/clustering/11_wine_mistral7b_inst_mntp_facet_colon_property_embeds_1inc_repeat10.json

echo 'Job Finished !!!'
