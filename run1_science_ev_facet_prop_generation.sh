#!/bin/bash --login

#SBATCH --job-name=l3ScEV

#SBATCH --output=logs/out_llama3_science_ev_facet_property.txt
#SBATCH --error=logs/err_llama3_science_ev_facet_property.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH --gres=gpu:1
#SBATCH -p gpu_v100
#SBATCH --exclusive

#SBATCH --mem=40G
#SBATCH -t 0-08:00:00


conda activate llm_prompts

CUDA_VISIBLE_DEVICES=0, python3 src/llama3_concept_facet_property_prompting_new.py --config_file configs/facet_prop_generation/llama3_science_ev_facet_property.json

echo 'Job Finished !!!'
