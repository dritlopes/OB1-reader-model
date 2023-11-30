import pandas as pd
import time
from types import SimpleNamespace
from utils import get_stimulus_text_from_file
import os


class TaskAttributes:
    """
    Class object that holds all parameters of task. Also permits to set default values.
    """

    def __init__(self, task_name, stim_name, stim, stimAll, language,
                 n_trials=0,
                 stimcycles=0,
                 blankscreen_type='blank', blankscreen_cycles_begin=0, blankscreen_cycles_end=0,
                 is_priming_task=False, ncyclesprime=0,
                 POS_implemented=False,
                 affix_implemented=False):
        self.task_name = task_name
        self.stim_name = stim_name
        self.stim = stim  # all stimuli data frame
        self.stim_all = stimAll  # list of stimuli
        self.n_trials = n_trials  # how many trials/texts from corpus/data should be used
        self.language = language
        self.stimcycles = stimcycles
        self.is_priming_task = is_priming_task
        self.blankscreen_type = blankscreen_type
        self.blankscreen_cycles_begin = blankscreen_cycles_begin
        self.blankscreen_cycles_end = blankscreen_cycles_end
        self.ncyclesprime = ncyclesprime
        self.POS_implemented = POS_implemented
        self.affix_implemented = affix_implemented
        self.totalcycles = self.blankscreen_cycles_begin + \
                           self.ncyclesprime + self.stimcycles + self.blankscreen_cycles_end


# NV: When designing a new task, set its attributes here. csv must contain a column called 'all', which contains all elements that are on screen during target presentation
# NV: function returns instance of TaskAttributes with corresponding attributes
def return_attributes(task_to_run, language, stimuli_filepath, file_separator):
    """" "return_attributes(task_to_run)" - Returns an instance of the TaskAttribute class. This class sets the atributes of the task, like stimuli,
 language and the number of stimulus cycles. Importantly, this function takes in an argument "task_to_run" which determines which task's attributes will be returned.
 For example: when the task is 'EmbeddedWords_German', the function reads a CSV file of the German stimuli, assigns it to the stim attribute of the TaskAttributes class,
 assigns 'German' to the language attribute and returns an instance of TaskAttributes with the associated attributes."""

    stim_data, stim_name = get_stimulus_text_from_file(stimuli_filepath, file_separator)

    if task_to_run == 'continuous_reading':
        stim_data['all'] = [text.encode("utf-8").decode("utf-8") for text in stim_data['all']]
        return TaskAttributes(
            task_to_run,
            stim_name,
            stim_data,
            stim_data['all'],
            language)

    elif task_to_run == 'EmbeddedWords':
        stim_data['stimulus'] = stim_data['stimulus'].astype('unicode')
        return TaskAttributes(
            task_to_run,
            stim_name,
            stim_data,
            list(stim_data['stimulus']),
            language='english',
            stimcycles=120,  # each stimulus was on screen for 3s
            is_priming_task=True,
            blankscreen_type='hashgrid',
            blankscreen_cycles_begin=5,  # blank screen before stimulus appears takes 200 ms  # FIXME : 20
            blankscreen_cycles_end=0,  # AL: no blank screen after stimulus appears?
            ncyclesprime=2,  # prime was on the screen for 50ms
            POS_implemented=False,
            affix_implemented=True
        )

    # KM: Adding EmbeddedWords_German
    elif task_to_run == 'EmbeddedWords_German':
        stim_data['stimulus'] = stim_data['stimulus'].astype('unicode')
        return TaskAttributes(
            task_to_run,
            stim_name,
            stim_data,
            list(stim_data['stimulus']),
            language='german',
            stimcycles=120,
            is_priming_task=True,
            blankscreen_type='hashgrid',
            blankscreen_cycles_begin=5,  # FIXME : 20
            blankscreen_cycles_end=0,
            ncyclesprime=2,
            POS_implemented=False,
            affix_implemented=True
        )

    elif task_to_run == 'Sentence':
        stim_data['stimulus'] = stim_data['stimulus'].astype('unicode')
        return TaskAttributes(
            task_to_run,
            stim_name,
            stim_data,
            list(stim_data['stimulus']),
            language='french',
            stimcycles=8,
            blankscreen_cycles_begin=8,
            blankscreen_cycles_end=16,
            POS_implemented=True,
            affix_implemented=True
        )

    elif task_to_run == 'Flanker':
        # NV: extra assignments needed for this task
        stim_data['stimulus'] = stim_data['stimulus'].astype('unicode')
        stim = stim_data[stim_data['condition'].str.startswith(('word'))].reset_index()
        return TaskAttributes(
            task_to_run,
            stim_name,
            stim,
            list(stim['stimulus']),
            language='french',
            stimcycles=6,
            blankscreen_cycles_begin=8,
            blankscreen_cycles_end=18,
            POS_implemented=False,
            affix_implemented=True
        )

    elif task_to_run == 'Transposed':
        stim_data['stimulus'] = stim_data['stimulus'].astype('unicode')
        return TaskAttributes(
            task_to_run,
            stim_name,
            stim_data,
            list(stim_data['stimulus']),
            language='french',
            is_priming_task=False,
            blankscreen_cycles_begin=8,
            blankscreen_type='fixation cross',
            stimcycles=120,
            POS_implemented=True,
            affix_implemented=True
        )

    elif task_to_run == 'Classification':
        stim_data['stimulus'] = stim_data['stimulus'].astype('unicode')
        return TaskAttributes(
            task_to_run,
            stim_name,
            stim_data,
            list(stim_data['stimulus']),
            language='dutch',
            is_priming_task=False,
            blankscreen_cycles_begin=8,
            blankscreen_type='fixation cross',
            stimcycles=7,  # Stimulus on screen for 170ms
            POS_implemented=True,
            affix_implemented=False
        )


# Control-flow parameters, that should be returned on a per-task basis
def return_task_params(task_attributes):
    """"return_task_params" - sets specific parameters for the task that is chosen to run,
    such as the time cycle of the task and other parameters such as bigram to word excitation, word inhibition, and attention.
    These parameters are set for experimentation and are established by the task_attributes object.
    """

    cycle_size = 25  # milliseconds that one model cycle is supposed to last (brain time, not model time)
    output_dir = time.time()

    # word activation
    bigram_to_word_excitation = 1.0  # inp. divded by #ngrams, so this param estimates excit per word [diff from paper] 1.65 for EmbeddedWords, 2.18 for classification and transposed
    bigram_to_word_inhibition = 0.0  # general inhibition on all words. The more active bigrams, the more general inhibition.
    word_inhibition = -2.5  # -.0018 (paper) # -2.5
    min_activity = 0.0
    max_activity = 1.0 # 1.0
    decay = -0.10 # -0.05 (paper) # -0.11 # AL: decay in word activation over time
    discounted_Ngrams = 5 # MM: Max extra wgt bigrams do to edges in 4-letter wrd w. gap 3. Added to bigram count in compute_input formula to compensate
    bigram_gap = 2  # How many in btw letters still lead to bigram? 5 (optimal) or 2 (paper, though there 3 because of different definition)
    # min_overlap = 0 # was 2 # min overlap for words to inhibit each other. MM: unnecessary, can be deleted later

    # threshold parameters
    max_threshold = 0.5  # mm: changed because max activity changed from 1.3 to 1
    freq_weight = 0.08 # NV: difference between max and min threshold # MM: words not in corpus have no freq, repaired by making freq less important
    word_length_similarity_constant = 0.15 # NV: determines how similar the length of 2 words must be for them to be recognised as 'similar word length'
    frequency_flag = True  # use word freq in threshold
    # use_grammar_prob = False  # True for using grammar probabilities, False for using cloze, overwritten by uniform_pred
    # grammar_weight = 0.5  # only used when using grammar_prob

    # pre-activation based on predictability
    prediction_flag = None  # cloze # uniform # grammar # language_model # None
    topk = 'all'  # in case of language model providing predictions, save only the k highest predictions
    pred_threshold = 0.01  # in case of language model providing predictions, save only the predictions above certain threshold
    pred_weight = 0.1  # scaling parameters in pre-activation formula

    # attention
    attend_width = 5.0  # 5.0 for natural reading # NV: was set to 15 for flanker, 20 for sentence and 3 for transposed
    max_attend_width = 5.0  # 5 in paper; MM: used in reading sim where attend_with is dynamic.
    min_attend_width = 3.0
    attention_skew = 3  # 1 equals symmetrical distribution # 4 (paper)
    letPerDeg = .3
    refix_size = 0.2  # during refix, how much do we jump?
    salience_position = 0.5  # 1.29 (paper)

    # saccade
    sacc_optimal_distance = 7  # 8.0 (optimal) # 7.0 (paper)
    saccErr_scaler = 0.2  # to determine avg error for distance difference
    saccErr_sigma = 0.17  # basic sigma
    saccErr_sigma_scaler = 0.06  # effect of distance on sigma
    mu, sigma = 12, 4  # 4.9, 2.2 (paper)
    recog_speeding = 5.0  # 1.1
    use_saccade_error = True

    # tuning
    include_sacc_type_sse = True  # Include the sse score based on the saccade type probability plot
    sacc_type_objective = "total"  # If "total" all subplots will be included in the final sse, single objectives can be "length", "freq" or "pred"
    include_sacc_dist_sse = True  # Include the SSE score derived from the saccade_distance.png plot
    tuning_measure = "SSE"  # can be "KL" or "SSE"
    discretization = "bin"  # can be "bin" or "kde"
    objective = []  # empty list for total SSE/KL, for single objectives: "total viewing time", "Gaze durations", "Single fixations", "First fixation duration", "Second fixation duration", "Regression"
    epsilon = 0.1  # Step-size for approximation of the gradient

    # specific experiment set-up
    trial_ends_on_key_press = False  # whether trial ends when word is recognized, or should keep going until end of cycle (3350 ms)

    # affix system
    affix_system, simil_algo, max_edit_dist, short_word_cutoff = False, '', 0, 0
    if task_attributes.affix_implemented:
        affix_system = True
        simil_algo = 'lcs'  # can be lev, lcs, startswith
        max_edit_dist = 1  # NV: maximum allowed distance between word and inferred stem, to be considered matching (relates to affix system)
        short_word_cutoff = 3

    # evaluation
    evaluation_measures = ['skip',
                           'single_fix',
                           'single_fix_duration',
                           'first_fix_duration',
                           'gaze_duration',
                           'total_reading_time',
                           'regression_in']
    fixed_factors = ['predictability']

    task_params = dict(locals())
    # NV: task_attributes is given as input, so would end up in the namespace if not removed.
    task_params.pop('task_attributes')

    return task_params


def return_params(global_params):
    # NV: fetch all attributes of the task to run, specified in parameters_exp. Creates an object
    task_attributes = return_attributes(global_params['task_to_run'], global_params['language'],
                                        global_params['stimuli_filepath'], global_params['stimuli_separator'])

    # NV: get parameters corresponding to type of given task. Returns dictionary
    task_params = return_task_params(task_attributes)

    # put all attributes of separate objects into one pm object
    pm = SimpleNamespace(**{**task_attributes.__dict__, **task_params, **global_params})

    return pm