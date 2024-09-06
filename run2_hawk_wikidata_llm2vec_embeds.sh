#!/bin/bash --login

#SBATCH --job-name=wikiembeds
#SBATCH --output=logs/wikidata/out_run2_hawk_wikidata_llm2vec_embeds.txt
#SBATCH --error=logs/wikidata/err_run2_hawk_wikidata_llm2vec_embeds.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1
#SBATCH --exclusive

#SBATCH --mem=64G
#SBATCH -t 0-6:00:00

conda activate brand_new

python src/embeds_llm2vec.py

echo 'Job Finished !!!'
