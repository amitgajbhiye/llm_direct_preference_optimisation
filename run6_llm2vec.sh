#!/bin/bash --login

#SBATCH -A scw1858
#SBATCH --job-name=llm2vec

#SBATCH --output=logs/out_llm2vec_embeds.txt
#SBATCH --error=logs/err_llm2vec_embeds.txt

#SBATCH --nodes=1
#SBATCH -p highmem

#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --mem=35G
#SBATCH -t 0-03:00:00

conda activate ven

python3 src/embeds_llm2vec.py

echo 'Job Finished !!!'
