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
# install extra packages
#pip install transformers
#pip install statsmodels
#pip install matplotlib
#pip install seaborn
#pip install sentencepiece


# cd to directory with program
cd $HOME/OB1-reader-model/src

# Run program
python $HOME/OB1-reader-model/scr/main.py $HOME/OB1-reader-model/data/processed/Provo_Corpus.csv --eye_tracking_filepath $HOME/OB1-reader-model/data/raw/Provo_Corpus-Eyetracking_Data.csv --results_identifier prediction_flag --experiment_parameters_filepath $HOME/OB1-reader-model/scr/experiment_parameters.json