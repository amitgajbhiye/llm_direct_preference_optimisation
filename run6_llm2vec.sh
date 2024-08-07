#!/bin/bash --login

#SBATCH -A scw1858
#SBATCH --job-name=llm2vec

#SBATCH --output=logs/out_llm2vec_embeds.txt
#SBATCH --error=logs/err_llm2vec_embeds.txt

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH -p gpu_v100
#SBATCH --mem=35G
#SBATCH --gres=gpu:1
#SBATCH -t 0-03:00:00

conda activate llm_prompts

huggingface-cli login --token "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"

python3 src/embeds_llm2vec.py

echo 'Job Finished !!!'
