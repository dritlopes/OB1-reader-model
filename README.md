# The OB1-reader 
### A model of word recognition, eye movements and now, comprehension during reading

by Adrielli Lopes & Martijn Meeter (Vrije Universiteit Amsterdam)

This repository contains the code to run the OB1-reader (first proposed by Snell et al. 2018), a model that simulates word recognition and eye movement control during reading. The model is a computational implementation of cognitive theories on reading and can be used to test predictions of these theories on various experimental set-ups. Previous studies have shown both quantitative and qualitative fits to human behavioural data in various tasks (e.g. Snell et al. 2018 for natural reading, Meeter et al. 2020 for flankers and priming).

The theoretical framework OB1-reader formulates primarily results from integrating relative letter-position coding (Grainger & Van Heuven, 2004) and parallel graded processing (e.g. Engbert et al. 2005) to bridge accounts on single-word recognition and eye movements during reading, respectively. These are largely based on bottom-up, automatic and early processing of visual input. Because language processing during reading is not limited to bottom-up processing and word recognition, we are currently working on expanding the model to contain feedback processes at the semantic and discourse levels to simulate reading comprehension in various dimensions: as a process, as a product and as a skill. This is the main goal of my PhD project entitled *Computational Models of Reading Comprehension.*

## Repository structure

### `/data`

This folder should contain all data needed for running simulations. It is made of four sub-folders:

* `/raw`: all corpus data (e.g. Provo (Luke & Christianson, 2018)) and resource data (e.g. SUBTLEX-UK (Van Heuven et al., 2014)) from other sources.
* `/processed`: all the pre-processed data to be used in the model simulations (e.g. Provo data cleaned and re-aligned; predictability values from the language models stored as a look-up dictionary).
* `/model_output`: all output of the model simulations (fixation-centered data).
* `/analysed`: all data resulting from analysing or evaluating the output of the model simulations.

### `/src`

This folder contains the scrips used to run the simulations. 

* `main.py`: this is the script called from the command line. It parses the command line arguments and calls the relevant functions according to the arguments given.
* `run_simulations.sh`: shell script to run simulations in a remote server with GPU memory. Useful if you are running a considerable amount of simulations per condition (e.g. 100).
* `parameters.py`: hyperparameters are set and returned, task attributes are constructed and returned.
* `simulate_experiment.py`: this is the core of the task simulation, where the main code for running the experiments is implemented.
* `reading_components.py`: the major functions corresponding to sub-processes in word recognition and eye movement are defined. These sub-processes include word activity computation, slot matching and saccade programming. *Currently in this repo only the task of natural reading is implemented. Other tasks to follow.*
* `reading_helper_functions.py`: contains helper functions for the reading simulation, e.g. calculating a letter's visual accuity.
* `utils.py`: contains helper functions for setting up the experiment run, e.g. reading in the stimulus file.
* `evaluation.py`: evaluates output of model simulations and makes plots.

## How to run the code

To run a reading simulation, enter the following arguments in the command line:
* `stimuli_filepath`: the path to the stimuli input to the model.
* `--task_to_run`: which reading task the model should perform. Default: continuous reading. *Currently only continuous reading implemented in this repo.*
* `--number_of_simulations`: how many simulations should the model run. Default: None. 
* `--language`: which language the stimulus is in. Default: English. Options: English, German, French and Dutch.
* `--run_exp`: should the model run stimulation(s)? Default: True.
* `--analyse_results`: should the output of the model be analysed directly after simulation? Default: False.
* `--results_filepath`: path to directory where the model output should be stored. Default: None.
* `--results_identifier`: if you are running experiments with conditions that differ in the model parameters, specify which parameter(s) for writing the files out with the specified parameter value in the file name.
* `--parameters_filepath`: path to file with the model parameters used for previously ran simulations. Needed if you don't run simulations, but the output of previously run simulations should be evaluated (--run_exp = False; --analyse_results = True).
* `--experiment_parameters_filepath`: path to json file specifying which model parameters should be overwritten and with which values. This is useful in case of running experiments with different set of model parameters.
* `--eye_tracking_filepath`: path to file with eye-tracking data to compare with the output of model simulations (i.e. for model evaluation).
* `--stimuli_separator`: which separator is used in the stimuli file. Default: \t.

For instance, to run a simulation of the task of natural reading, with the stimulus stored in the path ../data/processed/PSC.txt and in the German language, the following should be entered in the command line:

`main.py ../data/processed/PSC.txt --task_to_run "continuous reading" --language german --run_exp True`

## References

Snell, J., van Leipsig, S., Grainger, J., & Meeter, M. (2018). OB1-reader: A model of word recognition and eye movements in text reading. Psychological review, 125(6), 969.
Meeter, M., Marzouki, Y., Avramiea, A. E., Snell, J., & Grainger, J. (2020). The role of attention in word recognition: Results from OB1‐reader. Cognitive science, 44(7), e12846.
Grainger, J., & Van Heuven, W. J. (2004). Modeling letter position coding in printed word perception.
Engbert, R., Nuthmann, A., Richter, E. M., & Kliegl, R. (2005). SWIFT: a dynamical model of saccade generation during reading. Psychological review, 112(4), 777.
Luke, S. G., & Christianson, K. (2018). The Provo Corpus: A large eye-tracking corpus with predictability norms. Behavior research methods, 50, 826-833.
Van Heuven, W. J., Mandera, P., Keuleers, E., & Brysbaert, M. (2014). SUBTLEX-UK: A new and improved word frequency database for British English. Quarterly journal of experimental psychology, 67(6), 1176-1190.