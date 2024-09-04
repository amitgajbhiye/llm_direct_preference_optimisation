#!/bin/bash --login

#SBATCH -A scw1858
#SBATCH --job-name=transAFP

#SBATCH --output=logs/ontology_clustering/out_run4_afp_mistral_transport_1inc_facet_property_clustering.sh
#SBATCH --error=logs/ontology_clustering/err_run4_afp_mistral_transport_1inc_facet_property_clustering.sh

#SBATCH --nodes=1
#SBATCH -p highmem

#SBATCH --ntasks=7
#SBATCH --ntasks-per-node=7
#SBATCH --cpus-per-task=1

#SBATCH --mem=120G
#SBATCH -t 0-10:00:00

conda activate venv

python3 src/facet_property_clustering.py --config_file configs/clustering/11_transport_mistral7b_inst_mntp_facet_colon_property_embeds_1inc_repeat10.json

echo 'Job Finished !!!'
