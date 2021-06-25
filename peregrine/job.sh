#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2G
#SBATCH --job-name=Neural Network Fractal Training

generate_data=false
train_neural_network=true

cd ../
module purge

if $generate_data
then 
    module load GCC/10.3.0
    cd data-generation/cpluplus_vViktor
    rm -r trainingData
    mkdir trainingData
    echo Starting Fractal Generation
    make run clean
    mv trainingData ../../nn/
    cd ../../
fi

if $train_neural_network
then
    module load Python/3.9.5-GCCcore-10.3.0
    module load matplotlib/3.2.1-foss-2020a-Python-3.8.2 
    module load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4
    module load CUDA/11.1.1-GCC-10.2.0
    cd nn/viktor
    echo Starting Neural Network Training
    python3 main.py # change to python
    cd ../../
fi