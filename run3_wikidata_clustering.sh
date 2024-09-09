#!/bin/bash --login

#SBATCH -A scw1858
#SBATCH --job-name=wikidata

#SBATCH --output=logs/wikidata/out_run3_wikidata_clustering.txt
#SBATCH --error=logs/wikidata/err_run3_wikidata_clustering.txt

#SBATCH --nodes=1
#SBATCH -p highmem

#SBATCH --ntasks=7
#SBATCH --ntasks-per-node=7
#SBATCH --cpus-per-task=1

#SBATCH --mem=75G
#SBATCH -t 0-01:00:00

conda activate venv

python3 src/wikidata_facet_property_clustering.py --config_file configs/clustering/13_all_wikidata_mistral7b_inst_mntp_facet_colon_property_embeds_1inc_repeat10.json

echo 'Job Finished !!!'
