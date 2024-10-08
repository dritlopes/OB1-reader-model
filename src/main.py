from datetime import datetime
import logging
import argparse
import time
from parameters import return_params
from simulate_experiment import simulate_experiment
from utils import write_out_simulation_data
import os
import pickle
from evaluation import evaluate_output
import json
from types import SimpleNamespace

# will create a new file everytime, stamped with date and time
now = datetime.now()
dt_string = now.strftime("_%Y_%m_%d_%H-%M-%S")
filename = f'logs/logfile{dt_string}.log'
if not os.path.isdir('logs'): os.mkdir('logs')
logging.basicConfig(filename=filename,
                    force=True,
                    encoding='utf-8',
                    level=logging.DEBUG,
                    format='%(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def simulate_reading(global_parameters):

    pm = return_params(global_parameters)
    logger.debug(pm)

    print("\nTASK: " + pm.task_to_run)
    print("\nLANGUAGE: " + pm.language)
    print("\nCORPUS: " + pm.stimuli_filepath)
    # print out the parameters you think are important for your experiment
    if pm.prediction_flag:
        print("\nPREDICTION_FLAG: " + str(pm.prediction_flag))
        print("\nPREDICTION_WEIGHT: " + str(pm.pred_weight))

    results_id = ''
    dir = f'../data/model_output/{dt_string}/'
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    if 'prediction_flag' in pm.results_identifier:
        pred_flag = pm.prediction_flag
        results_id = f'{pred_flag}_{pm.pred_weight}'
        if not pred_flag:
            pred_flag = 'baseline'
            results_id = f'{pred_flag}'

    if not pm.results_filepath:
        pm.results_filepath = f"{dir}simulation_{pm.stim_name}_{pm.task_to_run}_{results_id}.csv"
    if not pm.parameters_filepath:
        pm.parameters_filepath = f"{dir}parameters_{pm.stim_name}_{pm.task_to_run}_{results_id}.pkl"

    start_time = time.perf_counter()
    # runs experiment
    simulation_data = simulate_experiment(pm)
    time_elapsed = time.perf_counter() - start_time
    print("Time elapsed: " + str(time_elapsed))
    # save out result fixations
    write_out_simulation_data(simulation_data, pm.results_filepath)
    # save out set of parameters from experiment
    prs = vars(pm)
    with open(pm.parameters_filepath, 'wb') as outfile:
        pickle.dump(prs, outfile)

    if pm.optimize:
        print(pm.optimize)
        print("Using: " + pm.tuning_measure)
        if any(pm.objective):
            print("Single Objective: " + pm.tuning_measure + " of " + pm.objective)
        else:
            print("Using total " + pm.tuning_measure)
        print("Step-size: " + str(pm.epsilon))
        print("-------------------")
        pass
        # epsilon = pm.epsilon
        # parameters, bounds, names = get_params(pm)
        # OLD_DISTANCE = np.inf
        # N_RUNS = 0
        # results = scipy.optimize.fmin_l_bfgs_b(func=reading_function,
        #                                        args=(names),
        #                                        x0=np.array(parameters),
        #                                        bounds=bounds,
        #                                        approx_grad=True, disp=True,
        #                                        epsilon=epsilon)
        # with open("results_optimization.pkl", "wb") as f:
        #     pickle.dump(results, f)
    return pm

def main():

    useparser=False
    if useparser:
        parser = argparse.ArgumentParser()
        parser.add_argument('stimuli_filepath')
        parser.add_argument('--stimuli_separator', default='\t')
        parser.add_argument('--task_to_run', default='continuous_reading')
        parser.add_argument('--language', default='english')
        parser.add_argument('--run_exp', default='True', help='Should the experiment simulation run?', choices=['True', 'False']),
        parser.add_argument('--number_of_simulations', default=None, help='How many times should I run a simulation?')
        parser.add_argument('--analyze_results', default="False", help='Should the results be analyzed?', choices=["True", "False"])
        parser.add_argument('--results_filepath', default=None, help='Path to file with results to be analysed if analyse_results=True and run_exp=False')
        parser.add_argument('--results_identifier', default=None, help='Which parameter/variable differ conditions')
        parser.add_argument('--parameters_filepath', default=None, help='Path to file with parameters of the results to be analysed if analyse_results=True and run_exp=False')
        parser.add_argument('--eye_tracking_filepath', default=None, help='If analyzing results, where are the observed values which the model output should be compared with?')
        parser.add_argument('--experiment_parameters_filepath', default=None, help='If you want to run different model setups at once, provide set of parameters for each model/experiment condition')
        parser.add_argument('--optimize', default="False", help='Should the parameters be optimized using evolutionary algorithms?', choices=["True", "False"])
        parser.add_argument('--print_process', default="False", choices=["True", "False"])
        parser.add_argument('--plotting', default='False', choices=['True', 'False'])

        args = parser.parse_args()
        global_parameters = {
            "task_to_run" : args.task_to_run,
            "stimuli_filepath": args.stimuli_filepath,
            "stimuli_separator": args.stimuli_separator,
            "language": args.language,
            "run_exp": eval(args.run_exp),
            "number_of_simulations": int(args.number_of_simulations),
            "analyze_results": eval(args.analyze_results),
            "results_filepath": args.results_filepath,
            "results_identifier": args.results_identifier,
            "parameters_filepath": args.parameters_filepath,
            "eye_tracking_filepath": args.eye_tracking_filepath,
            "experiment_parameters_filepath": args.experiment_parameters_filepath,
            "optimize": eval(args.optimize),
            "print_process": eval(args.print_process),
            "plotting": eval(args.plotting)
        }
    else:
        global_parameters = {
            "task_to_run": 'continuous_reading', # Flanker
            "stimuli_filepath": "../data/processed/Provo_Corpus.csv", # "../data/raw/stimuli_en.csv"
            "stimuli_separator": "\t",
            "language": 'english',
            "run_exp": False,
            "analyze_results": True,
            "results_filepath": "",
            "parameters_filepath": "",
            # "results_filepath": ["../data/model_output/_2023_12_07_22-32-42/simulation_Provo_Corpus_continuous_reading_cloze_0.1.csv",
            #                      "../data/model_output/_2023_12_07_22-32-42/simulation_Provo_Corpus_continuous_reading_gpt2_0.1.csv",
            #                      "../data/model_output/_2023_12_07_22-32-42/simulation_Provo_Corpus_continuous_reading_llama_0.1.csv"],
            # "parameters_filepath": ["../data/model_output/_2023_12_07_22-32-42/parameters_Provo_Corpus_continuous_reading_cloze_0.1.pkl",
            #                         "../data/model_output/_2023_12_07_22-32-42/parameters_Provo_Corpus_continuous_reading_gpt2_0.1.pkl",
            #                         "../data/model_output/_2023_12_07_22-32-42/parameters_Provo_Corpus_continuous_reading_llama_0.1.pkl"],
            "eye_tracking_filepath": '', # '../data/raw/Provo_Corpus-Eyetracking_Data.csv',
            "results_identifier": 'prediction_flag', # 'prediction_flag',
            "experiment_parameters_filepath": '', # 'experiment_parameters.json',
            "optimize": False,
            "print_process": False,
            "plotting": False
        }

    # register set of parameters of each experiment.
    all_parameters = []

    if global_parameters["run_exp"]:
        # needed if more than one experiment (set of model of parameters) to be run at once.
        if global_parameters['experiment_parameters_filepath']:
            with open(global_parameters['experiment_parameters_filepath']) as infile:
                exp_prs = json.load(infile)
            model_instances = exp_prs['parameters']
            for instance in model_instances:
                global_parameters.update(instance)
                all_parameters.append(simulate_reading(global_parameters))
        # if just one set of parameters, defined in parameters.py (experiment_parameters_filepath not given)
        else:
            all_parameters.append(simulate_reading(global_parameters))

    # evaluate all models against human data
    if global_parameters["analyze_results"]:
        # if simulations weren't run, i.e. results were already given and just need to be analysed
        if not global_parameters["run_exp"]:
            # it needs a parameters filepath
            try:
                # makes sure having more than one set of results to analyse is also possible
                if type(global_parameters["parameters_filepath"]) != list:
                    global_parameters["parameters_filepath"] = global_parameters["parameters_filepath"].split()
                for parameters_filepath in global_parameters["parameters_filepath"]:
                    with open(parameters_filepath, 'rb') as infile:
                        parameters = pickle.load(infile)
                        all_parameters.append(SimpleNamespace(**{**parameters}))
            except FileNotFoundError:
                print("Parameter filepath not found. Please give at least one filepath to the set of parameters of the results you want to analyse, by filling in the terminal argument --parameters_filepath. The file should be in pickle format.")
        evaluate_output(all_parameters)


if __name__ == '__main__':
    main()

