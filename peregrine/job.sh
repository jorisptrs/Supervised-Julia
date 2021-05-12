#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2G
#SBATCH --job-name=Neural Network Training

module purge
module load Python/3.6.4-foss-2018a
module load matplotlib/2.1.2-foss-2018a-Python-3.6.4

echo Overview of modules that are loaded
module list

echo Starting Python program
python script.py /data_dir