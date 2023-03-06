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
                 stimcycles=0,
                 blankscreen_type='blank', blankscreen_cycles_begin=0, blankscreen_cycles_end=0,
                 is_priming_task=False, ncyclesprime=0,
                 POS_implemented=False,
                 affix_implemented=False):
        self.task_name = task_name
        self.stim_name = stim_name
        self.stim = stim
        self.stim_all = stimAll
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
def return_attributes(task_to_run, language, stimuli_filepath):

    """" "return_attributes(task_to_run)" - Returns an instance of the TaskAttribute class. This class sets the atributes of the task, like stimuli,
 language and the number of stimulus cycles. Importantly, this function takes in an argument "task_to_run" which determines which task's attributes will be returned.
 For example: when the task is 'EmbeddedWords_German', the function reads a CSV file of the German stimuli, assigns it to the stim attribute of the TaskAttributes class,
 assigns 'German' to the language attribute and returns an instance of TaskAttributes with the associated attributes."""

    stim_data = get_stimulus_text_from_file(stimuli_filepath)
    stim_name = os.path.basename(stimuli_filepath).replace('.txt', '').replace('.csv', '')

    if task_to_run == 'continuous reading':
        stim_data['all'] = stim_data['all'].encode("utf-8").decode("utf-8")
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
            stimcycles=120, # each stimulus was on screen for 3s
            is_priming_task=True,
            blankscreen_type='hashgrid',
            blankscreen_cycles_begin=5, # blank screen before stimulus appears takes 200 ms  # FIXME : 20
            blankscreen_cycles_end=0, #AL: no blank screen after stimulus appears?
            ncyclesprime=2, # prime was on the screen for 50ms
            POS_implemented = False, 
            affix_implemented = True
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
            POS_implemented = False, 
            affix_implemented = True 
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
            POS_implemented = True, 
            affix_implemented = True
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
            POS_implemented = False, 
            affix_implemented = True
        )

    elif task_to_run == 'Transposed':
        stim_data['stimulus'] = stim_data['stimulus'].astype('unicode')
        return TaskAttributes(
            task_to_run,
            stim_name,
            stim_data,
            list(stim_data['stimulus']),
            language='french',
            is_priming_task = False,
            blankscreen_cycles_begin = 8,
            blankscreen_type='fixation cross',
            stimcycles = 120,
            POS_implemented = True, 
            affix_implemented = True
        )

    elif task_to_run == 'Classification':
        stim_data['stimulus'] = stim_data['stimulus'].astype('unicode')
        return TaskAttributes(
            task_to_run,
            stim_name,
            stim_data,
            list(stim_data['stimulus']),
            language='dutch',
            is_priming_task = False,
            blankscreen_cycles_begin = 8,
            blankscreen_type='fixation cross',
            stimcycles = 7,  # Stimulus on screen for 170ms
            POS_implemented = True, 
            affix_implemented = False
        )


# Control-flow parameters, that should be returned on a per-task basis
def return_task_params(task_attributes):

    """"return_task_params" - sets specific parameters for the task that is chosen to run,
    such as the time cycle of the task and other parameters such as bigram to word excitation, word inhibition, and attention.
    These parameters are set for experimentation and are established by the task_attributes object.
    """

    cycle_size = 25 # milliseconds that one model cycle is supposed to last (brain time, not model time)
    output_dir = time.time()

    # word activation
    bigram_to_word_excitation = 1  # 0.0044 # 3.09269333333 # 2.18 # inp. divded by #ngrams, so this param estimates excit per word [diff from paper] 1.65 for EmbeddedWords, 2.18 for classification and transposed
    bigram_to_word_inhibition = 0.0 # -0.20625  # -0.65835 # -0.55  # general inhibition on all words. The more active bigrams, the more general inhibition.
    word_inhibition = -0.005 # -0.01  # -0.016093 # -0.002
    min_activity = 0.0
    max_activity = 1.0 # 1.3
    decay = -0.06 # -0.08 # AL: decay in word activation over time
    discounted_Ngrams = 7 # MM: Max extra wgt bigrams do to edges in 4-letter wrd w. gap 3. Added to bigram count in compute_input formula to compensate
    bigram_gap = 2  # How many in btw letters still lead to bigram? 5 (optimal) or 2 (paper, though there 3 because of different definition)
    min_overlap = 2 # min overlap for words to inhibit each other

    # attention
    attend_width = 5.0 # 8.0  # NV: #!!!: was set to 15 for flanker, 20 for sentence and 3 for transposed
    max_attend_width = 5.0 # AL: maybe increase this for reading simulation?
    min_attend_width = 3.0
    attention_skew = 4 # 1 equals symmetrical distribution # 4 (paper)
    letPerDeg = .3
    refix_size = 0.2
    salience_position = 5  # 5 (optimal) # 1.29 (paper)

    # saccade
    include_sacc_type_sse = True  # Include the sse score based on the saccade type probability plot
    sacc_type_objective = "total"  # If "total" all subplots will be included in the final sse, single objectives can be "length", "freq" or "pred"
    include_sacc_dist_sse = True  # Include the SSE score derived from the saccade_distance.png plot
    sacc_optimal_distance = 8 # 9.99  # 3.1 # 7.0 # 8.0 (optimal) # 7.0 (paper)
    saccErr_scaler = 0.2  # to determine avg error for distance difference
    saccErr_sigma = 0.17  # basic sigma
    saccErr_sigma_scaler = 0.06  # effect of distance on sigma

    # tuning
    tuning_measure = "SSE"  # can be "KL" or "SSE"
    discretization = "bin"  # can be "bin" or "kde"
    objective = []  # empty list for total SSE/KL, for single objectives: "total viewing time", "Gaze durations", "Single fixations", "First fixation duration", "Second fixation duration", "Regression"
    epsilon = 0.1  # Step-size for approximation of the gradient

    # model settings
    frequency_flag = True  # use word freq in threshold
    prediction_flag = True
    similarity_based_recognition = True
    use_saccade_error = True
    use_attendposition_change = True  # attend width influenced by predictability next wrd
    visualise = False
    slow_word_activity = False
    pauze_allocation_errors = False
    use_boundary_task = False
    corpora_repeats = 0  # how many times should corpus be repeated? (simulates diff. subjects)

    # fixation durations
    mu, sigma = 10, 4  # 4.9, 2.2 (paper)
    recog_speeding = 5.0  # 1.1 Decrease in av. SRT when word recognized

    # threshold parameters
    max_threshold = 0.7 # mm: changed because max activity changed from 1.3 to 1
    # MM: a number of words have no freq because not in corpus, repaired by making freq less important
    wordfreq_p = 0.2  # 0.2 #NV: difference between max and min threshold
    wordpred_p = 0.2  # Currently not used
    word_length_similarity_constant = 0.15 # 0.35  # 0.15 # NV: determines how similar the length of 2 words must be for them to be recognised as 'similar word length'
    use_grammar_prob = False  # True for using grammar probabilities, False for using cloze, overwritten by uniform_pred
    uniform_prob = False  # Overwrites cloze/grammar probabilities with 0.25 for all words
    grammar_weight = 0.5  # only used when using grammar_prob
    linear = False

    # experiment set-up
    trial_ends_on_key_press = False  # whether trial ends when word is recognized, or should keep going until end of cycle (3350 ms)

    # affix system
    affix_system, simil_algo, max_edit_dist, short_word_cutoff = False,'', 0, 0
    if task_attributes.affix_implemented:
        affix_system = True
        simil_algo = 'lcs'  # can be lev, lcs, startswith
        max_edit_dist = 1 # NV: maximum allowed distance between word and inferred stem, to be considered matching (relates to affix system)
        short_word_cutoff = 3

    task_params = dict(locals())
    # NV: task_attributes is given as input, so would end up in the namespace if not removed.
    task_params.pop('task_attributes')

    return task_params

def return_params(global_params):

    # NV: fetch all attributes of the task to run, specified in parameters_exp. Creates an object
    task_attributes = return_attributes(global_params['task_to_run'],global_params['language'],global_params['stimuli_filepath'])
    
    # NV: get parameters corresponding to type of given task. Returns dictionary
    task_params = return_task_params(task_attributes)
    
    # put all attributes of separate objects into one pm object
    pm = SimpleNamespace(**{**global_params, **task_attributes.__dict__, **task_params})

    return pm