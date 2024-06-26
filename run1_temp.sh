#!/bin/bash --login

#SBATCH --job-name=temp

#SBATCH --output=logs/out_llama3_temp.txt
#SBATCH --error=logs/err_llama3_temp.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p highmem

#SBATCH --mem=5G
#SBATCH -t 0-01:00:00

conda activate venv

python3 src/llama3_concept_facet_property_prompting_new.py --config_file configs/facet_prop_generation/llama3_science_ev_facet_property.json

echo 'Job Finished !!!'
