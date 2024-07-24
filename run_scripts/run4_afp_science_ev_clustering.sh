#!/bin/bash --login

#SBATCH --job-name=afp_env

#SBATCH --output=logs/out_run4_afp_science_ev_clustering.txt
#SBATCH --error=logs/err_run4_afp_science_ev_clustering.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p highmem

#SBATCH --mem=60G
#SBATCH -t 0-03:00:00

conda activate venv

python3 src/facet_property_clustering.py --config_file configs/clustering/bienc_science_ev.json

echo 'Job Finished !!!'
