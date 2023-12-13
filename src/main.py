from datetime import datetime
import logging
import argparse
import time
from parameters import return_params
from simulate_experiment import simulate_experiment, simulate_experiment_specific_tasks
from utils import write_out_simulation_data
import os
import pickle
from evaluation import evaluate_output
import json
from types import SimpleNamespace
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QComboBox, QLineEdit, QLabel, QWidget, QTextEdit, QCheckBox
from PyQt5.QtCore import Qt
import webbrowser
import sys
from logging.handlers import RotatingFileHandler
from types import SimpleNamespace
from contextlib import contextmanager
from PyQt5.QtCore import QThread, pyqtSignal
import json
import signal
import traceback

from utils_specific_tasks import write_out_simulation_data_specific


def on_experiment_finished(self):
    """
    This function is called when the experiment running in a separate thread is completed. 
    Its main role is to handle the post experiment activities. Like saving the configuration.
    """
    print("Experiment completed.")
    self.print_to_terminal("GUI: Experiment completed.")
    self.save_last_run_configuration(self.last_run_parameters)

class QTextEditLogger:
    """
    This class is designed to redirect standard output (sys.stdout) to a PyQt text widget. (If need be) 
    This is useful for capturing print statements and displaying them in the GUI.
    """
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.original_stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout

    def write(self, message):
        self.text_widget.append(message)

    def flush(self):
        pass

class ExperimentThread(QThread):
    """
    subclass of QThread and is designed to run an experiment in a separate thread to avoid blocking the main GUI thread.
    - When ran, it takes parameters required to run the experiment, inside simulate_reading is called with the provided parameters. 
    If an exception occurs during the simulation, it is caught and printed to the console. 

    Why Threading? - allows for running the simulation without freezing the GUI, if they both run in main thread, GUI would be completely unresponsive. 
    
    """

    # 
    finished = pyqtSignal()

    def __init__(self, global_parameters, sim_params, outfile_sim_data, outfile_skipped):
        super().__init__()
        self.global_parameters = global_parameters
        self.sim_params = sim_params
        self.outfile_sim_data = outfile_sim_data
        self.outfile_skipped = outfile_skipped

    def run(self):
        try:
            simulate_reading(self.global_parameters, self.sim_params, self.outfile_sim_data, self.outfile_skipped)
        except Exception as e:
            print("Error during simulation:", e)
            traceback.print_exc()
        self.finished.emit()


# Date and log file name
"""
Logging and time set up
"""
now = datetime.now()
dt_string = now.strftime("_%d_%m_%Y_%H-%M-%S")
filename = f'logs/logfile{dt_string}.log'
if not os.path.isdir('logs'): os.mkdir('logs')
logging.basicConfig(handlers=[RotatingFileHandler(filename, mode='w', backupCount=10)],
                    force=True,
                    level=logging.DEBUG,
                    format='%(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def simulate_reading(global_sim_params, sim_params, outfile_sim_data, outfile_skipped=None):
    """
    This function is designed to simulate a task based on the given parameters, 
    and subsequently save the results of the simulation to specified files.

    """
    # Convert global simulation parameters into a usable format.
    pm = return_params(global_sim_params)
    # Log the converted parameters for debugging.
    logger.debug(pm)

    # Print basic information about the task being simulated.
    print("\nTASK: " + pm.task_to_run)
    print("\nLANGUAGE: " + pm.language)
    print("\nCORPUS: " + pm.stimuli_filepath)

    # If prediction is a part of the simulation, print related.
    if pm.prediction_flag:
        print("\nPREDICTION_FLAG: " + str(pm.prediction_flag))
        print("\nPREDICTION_WEIGHT: " + str(pm.pred_weight))

    # Initialize an identifier for the results and directory for output.
    results_id = ''
    dir = f'data/model_output/{dt_string}/'

    # Create the directory for output if it does not exist.
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Set up the results identifier based on prediction flag.
    if 'prediction_flag' in pm.results_identifier:
        pred_flag = pm.prediction_flag
        # TODO test this
        if pred_flag == '':
            pred_flag = 'baseline'
        results_id = f'{pred_flag}_{pm.pred_weight}'
    
    # Define file paths for saving simulation results and parameters if not already set.
    if not pm.results_filepath:
        pm.results_filepath = f"{dir}simulation_{pm.stim_name}_{pm.task_to_run}_{results_id}.csv"
    if not pm.parameters_filepath:
        pm.parameters_filepath = f"{dir}sim_params_{pm.stim_name}_{pm.task_to_run}_{results_id}.pkl"

    # Record the start time of the simulation.
    start_time = time.perf_counter() 

    # Run the experiment based on the type of task specified in sim_params.
    # Code will execute if run_exp is there and runs smoothly. Useful way to check where things were going wrong.
    # This block is skipepd in run_exp is False
    if sim_params.get('run_exp', False):
        if sim_params.get('task_to_run') == 'continuous_reading':
            print("Before calling simulate_experiment")
            simulation_data, skipped_words = simulate_experiment(pm)
            print("After calling simulate_experiment")
        elif sim_params.get('task_to_run') == 'EmbeddedWords':
            print("Before calling simulate_experiment_specific_tasks")
            simulation_data, skipped_words = simulate_experiment_specific_tasks(pm)
            print("After calling simulate_experiment_specific_tasks")
        elif sim_params.get('task_to_run') == 'EmbeddedWords_German':
            print("Before calling simulate_experiment_specific_tasks")
            simulation_data, skipped_words = simulate_experiment_specific_tasks(pm)
            print("After calling simulate_experiment_specific_tasks")
        else:
            raise ValueError(f"Unknown task: {sim_params.get('task_to_run')}")

        # Save the simulation data and skipped words to files.
        with open(outfile_sim_data, "wb") as all_data_file:
            pickle.dump(simulation_data, all_data_file)
        if outfile_skipped:
            with open(outfile_skipped, "wb") as skipped_file:
                pickle.dump(skipped_words, skipped_file)

        # Print the simulation parameters and the elapsed time.
        print("Simulation parameters:", sim_params)
        time_elapsed = time.perf_counter() - start_time
        print("Time elapsed: " + str(time_elapsed))
        # save out result fixations
        if sim_params.get('task_to_run') == 'continuous_reading':
            write_out_simulation_data(simulation_data, pm.results_filepath)
        elif sim_params.get('task_to_run') == 'EmbeddedWords' or 'EmbeddedWords_German':
            write_out_simulation_data_specific(simulation_data, pm.results_filepath)

    # Placeholder for result analysis, if needed.
    if sim_params.get('analyze_results', False):
        pass
        # get_results_simulation(task, output_file_all_data,
        #                        output_file_unrecognized_words)

    # Placeholder for optimization, if needed.
    if sim_params.get('optimize', False):
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

# Application Window
class AppWindow(QMainWindow):
    """
    class defines the main window of the GUI application for the OB-1 Reader Model simulation.
    It initializes and sets up the user interface, manages user interactions, and handles the execution of simulation tasks.
    - It sets a variable for storing the parameters of the last run and calls initUI() to set up the user interface.
    - layout of the application: the title, description, dropdown menus, buttons, checkboxes, and text output area.

    AppWindow acts as the central hub, placing user input, simulation control, and output display.
    - The use of threads is for seamless and non-glitchy application of the GUI and the run of simulations.
    """
    def __init__(self):
        super().__init__()
        self.last_run_parameters = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Experiment Simulation')
        self.setGeometry(100, 100, 800, 600)

        # Create a container widget and layout
        mainWidget = QWidget(self)
        self.setCentralWidget(mainWidget)
        layout = QVBoxLayout(mainWidget)

        # Title and Description
        title = QLabel("OB-1 Reader Model", self)
        title_font = title.font()  
        title_font.setPointSize(18)  
        title.setFont(title_font)  
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        description = """OB1 is a reading-model that simulates the cognitive processes behind reading. For more information about the theory behind OB1 and how it works see:"""
        description_label = QLabel(description, self)
        description_label.setWordWrap(True)
        layout.addWidget(description_label)

        # Link to Research Paper
        research_paper_button = QPushButton('Link to OB-1 Research Paper', self)
        research_paper_button.clicked.connect(self.open_research_paper)
        layout.addWidget(research_paper_button)

        # Task Dropdown with default options
        task_label = QLabel("Select Task:")
        layout.addWidget(task_label)
        self.task_menu = QComboBox(self)
        self.task_menu.addItems(['continuous_reading', 'EmbeddedWords', 'EmbeddedWords_German'])
        self.task_menu.currentIndexChanged.connect(self.update_defaults_based_on_task)
        layout.addWidget(self.task_menu)

        # File path entry
        self.results_filepath_entry = QLineEdit(self)
        layout.addWidget(self.results_filepath_entry)
        self.parameters_filepath_entry = QLineEdit(self)
        layout.addWidget(self.parameters_filepath_entry)

        # Stimuli Filepath Dropdown with default options
        stimuli_label = QLabel("Select Stimuli Filepath:")
        layout.addWidget(stimuli_label)
        self.stimuli_filepath_menu = QComboBox(self)
        self.stimuli_filepath_menu.addItems([
            "data\processed\Provo_Corpus.csv",
            "stimuli\EmbeddedWords_stimuli_all_csv.csv",
            "stimuli\PSC_test.txt",
            "stimuli\EmbeddedWords_Nonwords_german_all_csv.csv"
        ])
        layout.addWidget(self.stimuli_filepath_menu)

        # Stimuli Separator
        self.stimuli_separator_entry = QLineEdit(self)
        self.stimuli_separator_entry.setText("\t")  # Set default value
        layout.addWidget(self.stimuli_separator_entry)

        # Language Dropdown with default options
        language_label = QLabel("Select Language:") 
        layout.addWidget(language_label) 
        self.language_menu = QComboBox(self)
        self.language_menu.addItems(['English', 'German', 'French', 'Dutch'])
        layout.addWidget(self.language_menu)

        # Results Identifier Dropdown with default options
        # CURRENTLY WILL BE HIDDEN
        self.results_identifier_menu = QComboBox(self)
        self.results_identifier_menu.addItems(['prediction_flag', 'Other Options Here'])
        layout.addWidget(self.results_identifier_menu)

        # Eye Tracking Filepath Dropdown with default options
        eye_tracking_label = QLabel("Select Eye Tracking Filepath:") 
        layout.addWidget(eye_tracking_label) 
        
        self.eye_tracking_filepath_menu = QComboBox(self)
        self.eye_tracking_filepath_menu.addItem('../data/raw/Provo_Corpus-Eyetracking_Data.csv')
        layout.addWidget(self.eye_tracking_filepath_menu)

        # Experiment Parameters filepath dropdown
        self.experiment_parameters_filepath_menu = QComboBox(self)
        self.experiment_parameters_filepath_menu.addItem(r'C:\Users\Konstantin\Documents\OB1-reader-model-master\src\experiment_parameters.json')
        layout.addWidget(self.experiment_parameters_filepath_menu)

        # Checkboxes for Boolean Options
        self.run_exp_checkbox = QCheckBox("Run Experiment", self)
        layout.addWidget(self.run_exp_checkbox)

        self.analyze_results_checkbox = QCheckBox("Analyze Results (not finished)", self)
        layout.addWidget(self.analyze_results_checkbox)

        self.optimize_checkbox = QCheckBox("Optimize (not finished)", self)
        layout.addWidget(self.optimize_checkbox)

        self.print_process_checkbox = QCheckBox("Print Process (not finished)", self)
        layout.addWidget(self.print_process_checkbox)

        self.plotting_checkbox = QCheckBox("Enable Plotting (not finished)", self)
        layout.addWidget(self.plotting_checkbox)

        # Hide empty lines for now
        self.results_filepath_entry.hide()  
        self.parameters_filepath_entry.hide()
        self.stimuli_separator_entry.hide()  
        self.results_identifier_menu.hide()
        self.experiment_parameters_filepath_menu.hide()

        # Set default
        self.task_menu.setCurrentIndex(self.task_menu.findText('continuous_reading')) 
        self.language_menu.setCurrentIndex(self.language_menu.findText('English'))
        self.results_identifier_menu.setCurrentIndex(self.results_identifier_menu.findText('prediction_flag'))
        self.run_exp_checkbox.setChecked(True)

        # Run Experiment Button
        run_button = QPushButton('Run Simulation', self)
        run_button.clicked.connect(self.run_experiment)
        layout.addWidget(run_button)

        # Run Previous Configuration Button
        run_prev_button = QPushButton('Run Previous Configuration', self)
        run_prev_button.clicked.connect(self.run_previous_experiment)
        layout.addWidget(run_prev_button)

        # Text Edit for Output (Terminal Window)
        self.output_terminal = QTextEdit(self)
        self.output_terminal.setReadOnly(True)  # Make the text area read-only
        layout.addWidget(self.output_terminal)


        self.show()

    def save_last_run_configuration(self, parameters):
        """
        Saves the configuration of the last run experiment to a file for future use.
        """
        with open('last_run_config.json', 'w') as file:
            json.dump(parameters, file)

    def run_experiment(self):
        """
        Handles the process of running a new experiment. 
        It gathers parameters from the GUI elements, updates the terminal with status messages and starts the experiment in a separate thread.

        """
        # Retrieve parameters from GUI elements
        task = self.task_menu.currentText()
        language = self.language_menu.currentText()
        stimuli_filepath = self.stimuli_filepath_menu.currentText()
        stimuli_separator = self.stimuli_separator_entry.text()
        results_identifier = self.results_identifier_menu.currentText()
        eye_tracking_filepath = self.eye_tracking_filepath_menu.currentText()
        experiment_parameters_filepath = self.experiment_parameters_filepath_menu.currentText()
        results_filepath = self.results_filepath_entry.text()
        parameters_filepath = self.parameters_filepath_entry.text()

        # Inform users to check the terminal for output
        self.print_to_terminal("Experiment started. Check terminal for detailed output.")

        # Retrieve boolean options from checkboxes
        run_exp = self.run_exp_checkbox.isChecked()
        analyze_results = self.analyze_results_checkbox.isChecked()
        optimize = self.optimize_checkbox.isChecked()
        print_process = self.print_process_checkbox.isChecked()
        plotting = self.plotting_checkbox.isChecked()
        

        # Prepare the global parameters dictionary
        global_parameters = {
            "task_to_run": task,
            "stimuli_filepath": stimuli_filepath,
            "stimuli_separator": stimuli_separator,
            "language": language,
            "run_exp": run_exp,
            "analyze_results": analyze_results,
            "results_identifier": results_identifier,
            "eye_tracking_filepath": eye_tracking_filepath,
            "experiment_parameters_filepath": experiment_parameters_filepath,
            "optimize": optimize,
            "print_process": print_process,
            "plotting": plotting
        }
        global_parameters['results_filepath'] = results_filepath if results_filepath else None
        global_parameters['parameters_filepath'] = parameters_filepath if parameters_filepath else None

        # Example simulation parameters
        sim_params = {
            "run_exp": self.run_exp_checkbox.isChecked(),
            "task_to_run": self.task_menu.currentText(),
            "analyze_results": self.analyze_results_checkbox.isChecked(),
            "optimize": self.optimize_checkbox.isChecked(),
            # Add other parameters as needed 
        }

        self.last_run_parameters = {'global': global_parameters, 'sim': sim_params}
        self.save_last_run_configuration(self.last_run_parameters)

        self.print_to_terminal(f"Selected Task: {task}")
        self.print_to_terminal("Experiment parameters set. Starting the experiment...")
        self.print_to_terminal("Please check the terminal for detailed output.")

        # Define output file paths
        outfile_sim_data = "path_to_simulation_data_file.pkl"
        outfile_skipped = "path_to_skipped_data_file.pkl"

        # Start the experiment in a separate thread
        self.experiment_thread = ExperimentThread(global_parameters, sim_params, outfile_sim_data, outfile_skipped)
        self.experiment_thread.finished.connect(self.on_experiment_finished)
        self.experiment_thread.start()

    def on_experiment_finished(self):
        """
        Gets triggered when the experiment thread completes. 
        It updates the GUI terminal with the experiment's completion status.
        """
        print("Experiment completed.")
        self.print_to_terminal("GUI: Experiment completed.")

    def run_previous_experiment(self):
        """
        Loads the last run configuration from a file and runs the experiment with those parameters
        """
        try:
            with open('last_run_config.json', 'r') as file:
                parameters = json.load(file)
                self.last_run_parameters = parameters

                self.print_to_terminal("Running experiment with previous configuration.")
                self.print_to_terminal("Please check the terminal for detailed output.")

                # Define output file paths (modify as per needs)
                outfile_sim_data = "path_to_simulation_data_file.pkl"
                outfile_skipped = "path_to_skipped_data_file.pkl"

                # Start the experiment in a separate thread
                self.experiment_thread = ExperimentThread(parameters['global'], parameters['sim'], outfile_sim_data, outfile_skipped)
                self.experiment_thread.finished.connect(self.on_experiment_finished)
                self.experiment_thread.start()
        except Exception as e:
            self.print_to_terminal(f"Error loading previous configuration: {e}")

    def print_to_terminal(self, message):
        """
        display messages on the GUI terminal.
        """
        self.output_terminal.append(message)

    def open_research_paper(self):
        """
        Opens a link to the OB-1 research paper.
        """
        webbrowser.open_new('https://www.ncbi.nlm.nih.gov/pubmed/30080066')

    def update_defaults_based_on_task(self):
        """
        Updates default selections in the GUI based on the chosen task. 
        For example: sets default language and stimuli filepath based on the selected task.
        """
        # Get the currently selected task
        selected_task = self.task_menu.currentText()

        if selected_task == 'EmbeddedWords':
            # language to English
            self.language_menu.setCurrentText('English')
            # stimuli file path to the Embedded Words stimuli
            self.stimuli_filepath_menu.setCurrentText("stimuli\EmbeddedWords_stimuli_all_csv.csv")
        elif selected_task == 'EmbeddedWords_German':
            self.language_menu.setCurrentText('German')
            self.stimuli_filepath_menu.setCurrentText("stimuli\EmbeddedWords_Nonwords_german_all_csv.csv")
        elif selected_task == 'continuous_reading':
            self.language_menu.setCurrentText('English')
            self.stimuli_filepath_menu.setCurrentText("data\processed\Provo_Corpus.csv")

# Main Function
def main():
    """
    The main() function serves as the entry point for the GUI application.
    - Sets up logging
    - makes instance of QApplication, initializes the AppWindow which sets up and displays the main GUI window.
    - Starts the application by calling app.exec_(). This line keeps the application running and responsive to user inputs until it is closed!
    """
    # Allows to use control-c to stop the run, but does not work with two threads
    # signal.signal(signal.SIGINT, signal.SIG_DFL)
    # Logging setup
    now = datetime.now()
    dt_string = now.strftime("_%d_%m_%Y_%H-%M-%S")
    filename = f'logs/logfile{dt_string}.log'
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    logging.basicConfig(filename=filename,
                        force=True,
                        encoding='utf-8',
                        level=logging.DEBUG,
                        format='%(name)s %(levelname)s:%(message)s')
    logger = logging.getLogger(__name__)

    # Initialize PyQt application
    app = QApplication(sys.argv)
    ex = AppWindow()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

