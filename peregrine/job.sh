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
    module load foss/2018a
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
    module load Python/3.6.4-foss-2018a
    module load matplotlib/2.1.2-foss-2018a-Python-3.6.4
    # add Cuda and pytorch modules
    cd nn/viktor
    echo Starting Neural Network Training
    python3 main.py # change to python
    cd ../../
fi