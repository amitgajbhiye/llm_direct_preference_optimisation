#!/bin/bash --login

#SBATCH --job-name=equipTaxo

#SBATCH --output=logs/facet_prop_generation/out_run1_llama38b_hawk_equipment_taxo_few_shot_prompting.txt
#SBATCH --error=logs/facet_prop_generation/err_run1_llama38b_hawk_equipment_taxo_few_shot_prompting.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1
##SBATCH --exclusive

#SBATCH --mem=16G
#SBATCH -t 0-01:00:00

conda activate llm_prompts

python3 src/concept_facet_property_prompting.py --config_file configs/facet_prop_generation/2_llama3_5inc_equipment_facet_property.json

echo 'Job Finished !!!'
