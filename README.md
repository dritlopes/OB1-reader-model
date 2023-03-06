# The OB1-reader 
### A model of word recognition, eye movements and now, comprehension during reading

by Adrielli Lopes & Martijn Meeter (Vrije Universiteit Amsterdam)

*WORK IN PROGRESS*

This repository contains the code to run the OB1-reader (first proposed by Snell et al. 2018), a model that simulates word recognition and eye movement control during reading. The model is a computational implementation of cognitive theories on reading and can be used to test predictions of these theories on various experimental set-ups. Previous studies have shown both quantitative and qualitative fits to human behavioural data (e.g. () for eye movement and () for N400 response) in various tasks (e.g. () for natural reading, () for lexical decision, () for flankers).

The theoretical framework OB1-reader formulates primarily results from integrating relative letter-position coding () and parallel graded processing () to bridge accounts on single-word recognition and eye movements during reading, respectively. These are largely based on bottom-up, automatic and early processing of visual input. Because language processing during reading is not limited to bottom-up processing and word recognition, we are currently working on expanding the model to contain feedback processes at the semantic and discourse levels to simulate reading comprehension in various dimensions: as a process, as a product and as a skill. This is the main goal of my PhD project entitled *Computational Models of Reading Comprehension.*

## Repository structure

### `/data`

This folder should contain the resources for frequency and predictability values, if used. Once you run the code, it will also contain a list of words in the lexicon, and dicts with a frequency value for each word in the lexicon and a predictability value for each word in the stimulus text, if frequency and predictability are used.

### `/stimuli`

Place your stimulus file in this folder. It can be in .txt format without new lines for natural reading, or in .csv format for controlled reading (shorter texts of one to few words presented in trials, without saccade programming). In the case of controlled reading, the stimulus of each trial should be in a column named 'all'. You can find in this folder one example of stimuli in .txt for natural reading and one example in .csv for controlled reading.

### `/src`

This folder contains the scrips used to run the simulations. 

* `main.py`: this is the script called from the command line. It parses the command line arguments and calls the relevant functions according to the arguments given.
* `parameters.py`: hyper-parameters are set and returned, task attributes are constructed and returned.
* `simulate_experiment.py`: this is the core of the task simulation. The major functions corresponding to sub-processes in word recognition and eye movement are defined. These sub-processes include word activity computation, slot matching and saccade programming. *Currently in this repo only the task of natural reading is implemented. Other tasks to follow.*
* `reading_functions.py`: contains helper functions for the reading simulation, e.g. calculating a letter's visual accuity.
* `utils.py`: contains helper functions for setting up the experiment run, e.g. reading in the stimulus file.
* `visualise_reading.py`: *not yet in this repo, to follow*
* `analyse_results.py`: *not yet in this repo, to follow*

### `/results`
This folder stores the outputs of model after a reading simulation.

## How to run the code

To run a reading simulation, enter the following arguments in the command line:
* `stimuli_filepath`: the path to the stimuli input to the model.
* `--task_to_run`: which reading task the model should perform. Default: continuous reading. *Currently only continuous reading implemented in this repo.*
* `--language`: which language the stimulus is in. Default: English. Options: English, German, French and Dutch.
* `--run_exp`: should the model run a stimulation? Default: True.
* `--analyse_results`: should the output of the model be analysed directly after simulation? Default: False. *Not yet implemented*
* `--optimize`: should the parameters be optimized using evolutionary algorithms? Default: False. *Not yet implemented.*
* `--print_stim`: option to print out stimulus on command line as it is input to the model. Default: False. *Not yet implemented.*
* `--plotting`: option to get plots of the results. Default: False. *Not yet implemented.*

For instance, to run a simulation of the task of natural reading, with the stimulus stored in the path ../stimuli/PSC.txt and in the German language, the following should be entered in the command line:

`main.py ../stimuli/PSC.txt --task_to_rum "continuous reading" --language german --run_exp True`

## References


