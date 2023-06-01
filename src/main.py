from datetime import datetime
import logging
import argparse
import time
from parameters import return_params
from simulate_experiment import simulate_experiment
from utils import write_out_simulation_data
import os
import pickle


def simulate_reading(parameters, outfile_sim_data, output_file_parameters, outfile_skipped=None):

    if parameters.run_exp:
        simulation_data, skipped_words = simulate_experiment(parameters)
        if outfile_sim_data:
            write_out_simulation_data(simulation_data, outfile_sim_data)
        if output_file_parameters:
            prs = vars(parameters)
            with open(output_file_parameters, 'wb') as outfile:
                pickle.dump(prs, outfile)
        if outfile_skipped:
            write_out_simulation_data(skipped_words, outfile_skipped, type='skipped')

    if parameters.analyze_results:
        pass
        # get_results_simulation(task, output_file_all_data,
        #                        output_file_unrecognized_words)

    if parameters.optimize:
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


def main():

    # will create a new file everytime, stamped with date and time. #TODO; build system to keep only last X logs
    now = datetime.now()
    dt_string = now.strftime("_%d_%m_%Y_%H-%M-%S")
    filename = f'logs/logfile{dt_string}.log'
    if not os.path.isdir('logs'): os.mkdir('logs')
    logging.basicConfig(filename=filename,
                        force=True,
                        encoding='utf-8',
                        level=logging.DEBUG,
                        format='%(name)s %(levelname)s:%(message)s')
    logger = logging.getLogger(__name__)

    useparser=False
    if useparser:
        parser = argparse.ArgumentParser()
        parser.add_argument('stimuli_filepath')
        parser.add_argument('--stimuli_separator',default='\t')
        parser.add_argument('--task_to_run',default='continuous reading')
        parser.add_argument('--language', default='english')
        parser.add_argument('--run_exp',default='True',help='Should the experiment simulation run?',choices=['True','False']),
        parser.add_argument('--number_of_simulations',default=1,help='How many times should I run a simulation?')
        parser.add_argument('--analyze_results',default="False",help='Should the results be analyzed?',choices=["True","False"])
        parser.add_argument('--optimize',default="False",help='Should the parameters be optimized using evolutionary algorithms?',choices=["True","False"])
        parser.add_argument('--print_stim',default="False",choices=["True","False"])
        parser.add_argument('--plotting',default='False',choices=['True','False'])

        args = parser.parse_args()
        global_parameters = {
            "task_to_run" : args.task_to_run,
            "stimuli_filepath": args.stimuli_filepath,
            "language": args.language,
            "run_exp": eval(args.run_exp),
            "number_of_simulations": eval(args.number_of_simulations),
            "analyze_results": eval(args.analyze_results),
            "optimize": eval(args.optimize),
            "print_stim": eval(args.print_stim),
            "plotting": eval(args.plotting)
        }
    else:
        global_parameters = {
            "task_to_run" : 'continuous reading', # EmbeddedWords # Flanker # continuous reading
            "stimuli_filepath": "../stimuli/Provo_Corpus.csv", # ../stimuli/EmbeddedWords_stimuli_all.csv #  "../stimuli/Flanker_stimuli_all.csv" # "../stimuli/PSC_test.txt"
            "stimuli_separator": "\t",
            "language": 'english', # english # french # german
            "run_exp": True,
            "number_of_simulations": 1,
            "analyze_results": False,
            "optimize": False,
            "print_stim": False,
            "plotting": False
        }

    pm = return_params(global_parameters)
    logger.debug(pm)

    print("\nTASK: " + pm.task_to_run)
    print("----PARAMETERS----")
    print("reading in " + pm.language)

    if pm.optimize:
        print(pm.optimize)
        print("Using: " + pm.tuning_measure)
        if any(pm.objective):
            print("Single Objective: " + pm.tuning_measure + " of " + pm.objective)
        else:
            print("Using total " + pm.tuning_measure)
        print("Step-size: " + str(pm.epsilon))
    print("-------------------")

    output_file_results = f"../results/simulation_{pm.stim_name}_{pm.task_to_run}_{dt_string}.csv"
    output_file_skipped = f"../results/skipped_words_{pm.stim_name}_{pm.task_to_run}_{dt_string}.csv"
    output_file_parameters = f"../results/parameters_{pm.stim_name}_{pm.task_to_run}_{dt_string}.pkl"

    start_time = time.perf_counter()
    simulate_reading(pm, output_file_results, output_file_parameters, output_file_skipped)
    time_elapsed = time.perf_counter() - start_time
    print("Time elapsed: " + str(time_elapsed))


if __name__ == '__main__':
    main()

