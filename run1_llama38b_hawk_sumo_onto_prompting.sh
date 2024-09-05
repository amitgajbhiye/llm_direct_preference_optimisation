#!/bin/bash --login

#SBATCH --job-name=sumoOnto
#SBATCH --output=logs/ontology_facet_prop/out_6_llama3_1inc_sumo_repeat_10.txt
#SBATCH --error=logs/ontology_facet_prop/err_6_llama3_1inc_sumo_repeat_10.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --gres=gpu:1
#SBATCH --exclusive

#SBATCH --mem=64G
#SBATCH -t 1-10:00:00

conda activate llm_vec

python3 src/concept_facet_property_prompting.py --config_file configs/facet_prop_generation/6_llama3_1inc_sumo_repeat_10.json

echo 'Job Finished !!!'
