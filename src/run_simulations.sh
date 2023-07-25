#!/bin/bash

# Set job requirements
#SBATCH -J run_OB1_simulations
#SBATCH -N 1
#SBATCH -p defq
#SBATCH --gpus=1
#SBATCH -t 2:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=a.t.lopesrego@vu.nl

# Environment modules
module load shared 2022
module load PyTorch/1.12.1-foss-2021a-CUDA-11.3.1

# cd to directory with program
cd $HOME/OB1-reader-model/src

# Run program
python main.py ../data/processed/Provo_Corpus.csv --eye_tracking_filepath ../data/raw/Provo_Corpus-Eyetracking_Data.csv --results_identifier prediction_flag --experiment_parameters_filepath experiment_parameters.json