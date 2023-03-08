from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import argparse
import time
import pickle
from parameters import return_params
from simulate_experiment import simulate_experiment
import os

def simulate_reading(parameters, outfile_sim_data, outfile_unrecognized):

    if parameters.run_exp:

        simulation_data = simulate_experiment(parameters)

        with open(outfile_sim_data, "wb") as all_data_file:
            pickle.dump(simulation_data, all_data_file)
        # with open(outfile_unrecognized, "wb") as unrecognized_file:
        #     pickle.dump(unrecognized_words, unrecognized_file)

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
    logging.basicConfig(handlers=[RotatingFileHandler(filename, mode='w', backupCount=10)],
                        force=True,
                        level=logging.DEBUG,
                        format='%(name)s %(levelname)s:%(message)s')
    logger = logging.getLogger(__name__)

    useparser=False
    if useparser:
        parser = argparse.ArgumentParser()
        parser.add_argument('stimuli_filepath')
        parser.add_argument('--task_to_run',default='continuous reading')
        parser.add_argument('--language', default='english')
        parser.add_argument('--run_exp',default='True',help='Should the experiment simulation run?',choices=['True','False'])
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
            "analyze_results": eval(args.analyze_results),
            "optimize": eval(args.optimize),
            "print_stim": eval(args.print_stim),
            "plotting": eval(args.plotting)
        }
    else:
        global_parameters = {
            "task_to_run" : 'continuous reading',
            "stimuli_filepath": "../stimuli/PSC_test.txt",
            "language": 'german',
            "run_exp": True,
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

    output_file_results, output_file_unrecognized_words = (
        "../results/results_" + pm.task_to_run + ".pkl", "../results/unrecognized_" + pm.task_to_run + ".pkl")

    start_time = time.perf_counter()
    simulate_reading(pm, output_file_results, output_file_unrecognized_words)
    time_elapsed = time.perf_counter() - start_time
    print("Time elapsed: " + str(time_elapsed))


if __name__ == '__main__':
    main()

