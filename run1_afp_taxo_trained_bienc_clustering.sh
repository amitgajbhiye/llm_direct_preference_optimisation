#!/bin/bash --login


#SBATCH -A scw1858
#SBATCH --job-name=commAFP

#SBATCH --output=logs/out_run1_afp_taxo_trained_bienc_clustering.txt
#SBATCH --error=logs/err_run1_afp_taxo_trained_bienc_clustering.txt

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p highmem
#SBATCH --cpus-per-task=30
#SBATCH --mem=75G
#SBATCH -t 0-11:00:00

conda activate venv

python3 src/facet_property_clustering.py --config_file configs/clustering/4_bienc_science_taxo_trained.json
python3 src/facet_property_clustering.py --config_file configs/clustering/4_bienc_food_taxo_trained.json
python3 src/facet_property_clustering.py --config_file configs/clustering/4_bienc_equipment_taxo_trained.json
python3 src/facet_property_clustering.py --config_file configs/clustering/4_bienc_environment_taxo_trained.json
python3 src/facet_property_clustering.py --config_file configs/clustering/4_bienc_commonsense_taxo_trained.json


echo 'Job Finished !!!'
