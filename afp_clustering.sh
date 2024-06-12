#!/bin/bash --login

#SBATCH --job-name=afp_cluster

#SBATCH --output=logs/out_afp_clustering.txt
#SBATCH --error=logs/err_afp_clustering.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p highmem

#SBATCH --mem=100G
#SBATCH -t 0-02:00:00

conda activate venv

python3 src/facet_property_bienc_embeds_cluster.py

echo 'Job Finished !!!'
