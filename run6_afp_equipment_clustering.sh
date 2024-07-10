#!/bin/bash --login

#SBATCH --job-name=afp_env

#SBATCH --output=logs/out_run6_afp_equipment_clustering.txt
#SBATCH --error=logs/err_run6_afp_equipment_clustering.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p highmem

#SBATCH --mem=70G
#SBATCH -t 0-03:00:00

conda activate venv

python3 src/facet_property_clustering.py --config_file configs/clustering/bienc_equipment.json

echo 'Job Finished !!!'
