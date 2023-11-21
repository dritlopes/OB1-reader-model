#!/bin/bash

# Set job requirements
#SBATCH -J run_OB1_simulations
#SBATCH -N 1
#SBATCH -p defq
#SBATCH --gpus=A30:1
#SBATCH -t 1:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=a.t.lopesrego@vu.nl

# Environment modules
module load shared 2022
module load PyTorch/1.12.1-foss-2021a-CUDA-11.3.1

# cd to directory with program
cd $HOME/OB1-reader-model/src

# Run program
for i in 0.05 0.1 0.2 do
  python main.py "../data/processed/Provo_Corpus.csv" --number_of_simulations 1 --experiment_parameters_filepath "experiment_parameters_$i.json" --analyze_results "True" --eye_tracking_filepath '../data/raw/Provo_Corpus-Eyetracking_Data.csv' --results_identifier 'prediction_flag' &
done
wait