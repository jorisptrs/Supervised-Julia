#!/bin/bash
#SBATCH --job-name=Neural Network Fractal Training
#SBATCH --partition=gpushort 
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

module purge
module load Python/3.9.5-GCCcore-10.3.0
module load matplotlib/3.2.1-foss-2020a-Python-3.8.2 
module load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4
module load scikit-learn/0.23.1-foss-2020a-Python-3.8.2
module load CUDA/11.1.1-GCC-10.2.0
module load Boost/1.66.0-foss-2018a-Python-3.6.4
module list

cd ../regression/cnn
echo Starting Neural Network Training
python main.py