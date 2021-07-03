#!/bin/bash
#SBATCH --job-name=Neural Network Fractal Training
#SBATCH --partition=gpushort 
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

module purge
module load Python/3.7.4-GCCcore-8.3.0
module load matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
module load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4
module load scikit-learn/0.22.2.post1-fosscuda-2019b-Python-3.7.4
module load CUDA/10.1.105-GCC-8.2.0-2.31.1
module list

cd ../regression/cnn
echo Starting Neural Network Training
python main.py