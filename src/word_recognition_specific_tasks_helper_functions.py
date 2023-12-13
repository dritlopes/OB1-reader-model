import numpy as np
import math


def initialize_trial(lexicon_size, pm, stim, trial):
    """
    Initialise trial-specific variables, such as eye and attention positions, 
    and various lexicon activity-related numpy arrays.
    
    Args:
    - lexicon_size (int): Size of the lexicon.
    - pm (object): Parameter object containing simulation settings.
    - stim (list): List of stimuli for the experiment.
    - trial (int): Current trial number.
    
    Returns:
    - tuple: Initialized trial-specific variables.
    """
    EyePosition = len(stim[trial]) // 2
    AttentionPosition = EyePosition
    lexicon_word_inhibition_np = np.zeros((lexicon_size), dtype=float)
    lexicon_total_input_np = np.zeros((lexicon_size), dtype=float)
    lexicon_word_activity_change = np.zeros((lexicon_size), dtype=float)
    lexicon_word_activity_np = np.zeros((lexicon_size), dtype=float)
    crt_word_activity_np = 0
    lexicon_word_activity_np[lexicon_word_activity_np < pm.min_activity] = pm.min_activity
    stimuli = []
    return EyePosition, AttentionPosition, lexicon_word_inhibition_np, lexicon_total_input_np, lexicon_word_activity_change, lexicon_word_activity_np, crt_word_activity_np, stimuli

def set_stimulus(trial, stim, pm, primes):
    """
    Set stimulus-related information for the current trial, such as the 
    padded stimulus, fixation point, and if applicable, the prime.

    Args:
    - trial (int): Current trial number.
    - stim (list): List of stimuli for the experiment.
    - pm (object): Parameter object containing simulation settings.
    - primes (list): List of primes for the priming task.
    
    Returns:
    - tuple: Stimulus-related information for the current trial.
    """
    stimulus = stim[trial]
    stimulus_padded = " " + stimulus + " "
    fixated_position_in_stim = math.floor(len(stimulus.split(' '))/2)
    eye_position = np.round(len(stimulus) // 2)
    attention_position = eye_position
    prime = None
    if pm.is_priming_task:
        prime = primes[trial]
        prime_padded = " " + prime + " "
    return stimulus, stimulus_padded, fixated_position_in_stim, eye_position, attention_position, prime, prime_padded

def handle_task_specifics(task, stimulus, stim, trial, all_data, pm, prime=None):
    """
    Extract and set the target word for the current trial based on the type of task.

    Args:
    - task (str): Type of task being run.
    - stimulus (str): Current stimulus.
    - stim (object): Stimulus dataframe.
    - trial (int): Current trial number.
    - all_data (list): List of trial-specific data.
    - pm (object): Parameter object containing simulation settings.
    
    Returns:
    - str: The target word for the current trial.
    """
    if task in ("Sentence", 'Classification', 'Transposed'):
        target = stimulus.split(" ")[stim['target'][trial]-1]  
        all_data[trial]['item_nr'] = stim['item_nr'][trial]
        all_data[trial]['position'] = stim['target'][trial]
        #all_data[trial]['POS'] = (POSdict[target] if pm.use_grammar_prob else None)

    elif task == "Flanker":
        target = (stimulus.split()[1] if len(stimulus.split()) > 1 else stimulus.split()[0])

    elif task in ("EmbeddedWords", "EmbeddedWords_German", "EmbeddedWords_French"):
        target = stim['target'][trial]
        all_data[trial]['prime'] = prime
        all_data[trial]['item_nr'] = stim['item_nr'][trial]

    return target

def handle_trial_task_specifics(trial, pm, stimulus, stim_df, prime, all_data):
    """
    Wrapper function to handle task specifics for the current trial.

    Args:
    - trial (int): Current trial number.
    - pm (object): Parameter object containing simulation settings.
    - stimulus (str): Current stimulus.
    - stim_df (object): Stimulus dataframe.
    - prime (str): Current prime.
    - all_data (list): List of trial-specific data.
    Returns:
    - str: The target word for the current trial.
    """
    target = handle_task_specifics(pm.task_to_run, stimulus, stim_df, trial, all_data, pm, prime)
    return target

def initialize_trial_data(trial, pm, stimulus, prime, stim_df, EyePosition, AttentionPosition):
    """
    Initialise a data dictionary for the current trial to store results and metrics.

    Args:
    - trial (int): Current trial number.
    - pm (object): Parameter object containing simulation settings.
    - stimulus (str): Current stimulus.
    - prime (str): Current prime.
    - stim_df (object): Stimulus dataframe.
    - EyePosition (int): Current eye position.
    - AttentionPosition (int): Current attention position.
    
    Returns:
    - dict: Initialized data dictionary for the current trial.
    """
    target = stim_df['target'][trial]
    condition = stim_df['condition'][trial]

    data = {
        'stimulus': stimulus,
        'prime': prime if pm.is_priming_task else None,
        'target': target,
        'condition': condition,
        'cycle': [],
        'lexicon activity per cycle': [],
        'lexicon activity squared': [],
        'stimulus activity per cycle': [],
        'target activity per cycle': [],
        'bigram activity per cycle': [],
        'top 10 bigrams': [],
        'bigrams ordered': [],
        'unit activations unordered': [],
        'ngrams': [],
        'exact recognized words positions': [],
        'exact recognized words': [],
        'eye position': EyePosition,
        'attention position': AttentionPosition,
        'word threshold': 0,
        'word frequency': 0,
        'word predictability': 0,
        'reaction time': [],
        'correct': [],
        'POS': [],
        'position': [],
        'inhibition_value': pm.word_inhibition,
        'wordlen_threshold': pm.word_length_similarity_constant,
        'target_inhib': [],
        'error_rate': 0,
    }
    return data

def reset_recognition_variables():
    """
    Reset recognition-related variables for the new trial, such as the current 
    cycle number and recognition flags.

    Returns:
    - tuple: Reset recognition-related variables for the new trial.
    """
    cycle_for_RT = 0
    cur_cycle = 0 
    recognized = False
    falseguess = False
    grammatical = False
    identif = False
    POSrecognition = {}
    # add here for more resets
    return cycle_for_RT, cur_cycle, recognized, falseguess, grammatical, identif, POSrecognition

def get_blankscreen_stimulus(blankscreen_type):
    """
    Returns the appropriate stimulus and padded stimulus based on the specified blank screen type.

    Parameters:
    - blankscreen_type (str): The type of blank screen stimulus to retrieve. Options are 'blank', 'hashgrid', and 'fixation cross'.

    Returns:
    - (str, str): A tuple containing the stimulus and its padded version.

    Example:
    For blankscreen_type='blank', the function returns ("", "  ").
    For blankscreen_type='hashgrid', the function returns ("#####", " ##### ").
    """
    if blankscreen_type == 'blank':  
        stimulus = ""
        stimulus_padded = "  "

    elif blankscreen_type == 'hashgrid':
        stimulus = "#####"  
        stimulus_padded = " ##### "

    elif blankscreen_type == 'fixation cross':
        stimulus = "+"
        stimulus_padded = " + "

    return stimulus, stimulus_padded

def is_similar_word_length(pm, len1, lengths_to_be_matched):
    """
    Checks if a given word length (len1) is similar to any word lengths in the provided list (lengths_to_be_matched). 
    A word length is considered similar if the difference between the two lengths is less than 15% of the length of the longer word.

    Parameters:
    - pm: contains model parameters, including the word_length_similarity_constant.
    - len1 (int): The word length to check.
    - lengths_to_be_matched (list of int): A list of word lengths to compare against.

    Returns:
    - bool: True if len1 is similar to any lengths in lengths_to_be_matched, False otherwise.
    """
    for len2 in lengths_to_be_matched:
        if abs(len1-len2) < (pm.word_length_similarity_constant * max(len1, len2)):
            result = True
        else:
            result = False
    return result

def slot_matching_order(stimulus):
    """
    Define the order in which words in a stimulus are checked, based on a static approach. 

    Args:
        stimulus (str): The stimulus string.

    Returns:
        order_match_check (list): List of indices representing the order in which words are checked.
    """
    
    # Check length of the stimulus, then determine the order in which words are matched to slots in the stimulus.
    # Words are checked in a pre-defined order based on the total number of words in the stimulus.
    n_words_in_stim = len(stimulus.split())
    
    if (n_words_in_stim < 2):
        # If stimulus has 1 word, it is checked first (note, indexing starts at 0!)
        order_match_check = [0]
    elif (n_words_in_stim == 2):
        # If stimulus has 2 words, first check right word, then left
        order_match_check = [1, 0]
    elif (n_words_in_stim == 3):
        # If stimulus has 3 words, first check middle word, then right, then left
        order_match_check = [1, 2, 0]
    elif (n_words_in_stim == 4):
        order_match_check = [2, 1, 3, 0]
    elif (n_words_in_stim == 5):
        order_match_check = [2, 3, 1, 4, 0]
    elif (n_words_in_stim == 6):
        order_match_check = [3, 2, 4, 1, 5, 0]
    elif (n_words_in_stim > 6):  # If more than 6 words, only consider first 7
        order_match_check = [3, 4, 2, 5, 1, 6, 0]
    
    return order_match_check

def all_ngrams_and_bigrams(all_ngrams):
    """
    Extracts and categorizes bigrams and monograms from a provided list of n-grams.

    Parameters:
    - all_ngrams (list of str): A list containing n-grams (combinations of characters).

    Returns:
    - (list of str, list of str, set of str): A tuple containing:
    - all_bigrams (list of str): All the bigrams (2-character combinations) extracted from the input n-grams.
    - all_monograms (list of str): All the monograms (1-character combinations) extracted from the input n-grams.
    - all_bigrams_set (set of str): A unique set of all the bigrams, ensuring no duplicates.

    Description:
    The function iterates through the provided list of n-grams. If an n-gram has a length of 2, it's considered a bigram and added to the bigrams list. If it has a length of 1, it's considered a monogram and added to the monograms list. Finally, a unique set of bigrams is returned to ensure no duplicates.
    """
    # Initialize lists to store extracted bigrams and monograms
    all_bigrams = []
    all_monograms = []
    # Iterate over the provided n-grams
    for ngram in all_ngrams:
        # If the n-gram has 2 characters, it's a bigram
        if len(ngram) == 2:
            all_bigrams.append(ngram)
        # If the n-gram has 1 character, it's a monogram
        else:
            all_monograms.append(ngram)
    # Create a set of bigrams to ensure unique values
    all_bigrams_set = set(all_bigrams)

    return all_bigrams, all_monograms, all_bigrams_set

def calculate_word_input(unitActivations, pm, allNgrams, lexicon, lexicon_word_bigrams, allBigrams_set, allMonograms, N_ngrams_lexicon):
    """
    Calculates the word input based on the activations of n-grams and various model parameters.

    Parameters:
    - unitActivations (dict): A dictionary mapping each n-gram to its activation value.
    - pm : containing model parameters, including 'bigram_to_word_inhibition'.
    - allNgrams (list of str): A list containing n-grams (combinations of characters).
    - lexicon (list of str): The lexicon containing all possible words.
    - lexicon_word_bigrams (list of lists): A list where each sublist contains the bigrams associated with a word from the lexicon.
    - allBigrams_set (set of str): A unique set of all bigrams.
    - allMonograms (list of str): A list containing all monograms.

    Returns:
    - word_input_np (numpy array): A numpy array representing the word input for each word in the lexicon.

    Description:
    The function computes the word input based on the activations of n-grams, 
    taking into account the number of n-grams and word-to-word inhibition parameters. 
    This input acts as a measure of the strength of evidence for each word in the lexicon based on the observed n-grams and their activations.
    """
    # Initialise word-to-word inhibition input
    wordBigramsInhibitionInput = 0
    # Calculate the inhibition input based on n-gram activations and the model's bigram to word inhibition parameter
    for ngram in allNgrams:
        wordBigramsInhibitionInput += pm.bigram_to_word_inhibition * \
            unitActivations[ngram]
    # Initialise a numpy array to store the word input for each word in the lexicon
    word_input_np = np.zeros(len(lexicon))

    # Compute excitatory and inhibitory input for each word in the lexicon
    for lexicon_ix, lexicon_word in enumerate(lexicon): 
        wordExcitationInput = 0

        # Calculate excitatory input based on overlapping bigrams with the observed n-grams
        bigram_intersect_list = allBigrams_set.intersection(lexicon_word_bigrams[lexicon_word])
        for bigram in bigram_intersect_list:
            wordExcitationInput += pm.bigram_to_word_excitation * unitActivations[bigram]
        
        # Calculate excitatory input based on overlapping monograms with the observed n-grams
        for monogram in allMonograms:
            if monogram in lexicon_word:
                wordExcitationInput += pm.bigram_to_word_excitation * unitActivations[monogram]

        # Combine excitatory and inhibitory input for the word
        word_input_np[lexicon_ix] = wordExcitationInput + wordBigramsInhibitionInput

    # Normalize the word input by dividing by the number of n-grams associated with each word
    word_input_np = word_input_np / np.array(N_ngrams_lexicon)

    return word_input_np

def match_words_in_slots(trial, lexicon_size, new_recognized_words, recognized, n_words_in_stim, order_match_check, stim_matched_slots, stimulus, above_thresh_lexicon_np, lexicon, 
                        affixes, lexicon_word_activity_np, pm, target, task, lexicon_index_dict, all_words_searched):
    """
    Matches recognized words to their respective slots in the stimulus based on various criteria.

    Parameters:
    - trial (int): The current trial number.
    - lexicon_size (int): The size of the lexicon.
    - new_recognized_words (np.array): A binary array indicating recognized words in the lexicon.
    - recognized (bool): Indicates whether the target word is recognized.
    - n_words_in_stim (int): The number of words in the stimulus.
    - order_match_check (list of int): The order in which slots in the stimulus should be checked.
    - stim_matched_slots (list of str): The words that are matched to each slot in the stimulus.
    - stimulus (str): The current stimulus.
    - above_thresh_lexicon_np (np.array): A binary array indicating words in the lexicon that are above the activation threshold.
    - lexicon (list of str): The lexicon containing all possible words.
    - affixes (list of str): The list of affixes.
    - lexicon_word_activity_np (np.array): An array representing the activity of each word in the lexicon.
    - pm: contains model parameters, including the word_length_similarity_constant.
    - target (str): The target word for the current trial.
    - task (str): The type of task being run.
    - lexicon_index_dict (dict): A dictionary mapping words to their respective index in the lexicon.
    - all_words_searched (list of str): A list of all the words that were searched in the lexicon so farr.

    Returns:
    - tuple: A tuple containing:
    - stim_matched_slots (list of str): The words that are matched to each slot in the stimulus.
    - new_recognized_words (np.array): Updated binary array indicating recognized words.
    - above_thresh_lexicon_np (np.array): Updated lexicon activity threshold array.
    - lexicon_word_activity_np (np.array): Updated lexicon word activity array.
    - recognized (bool): Updated recognition status for the target word.
    - falseguess (bool): Indicates if a false guess was made.
    - POSrecognition (list of str): Part-of-speech recognition results for each slot.
    - noun_count (int): Count of recognized nouns.
    - ver_count (int): Count of recognized verbs.
    - all_words_searched (list of str): All the words that were searched in the lexicon.

    Description:
    The function attempts to match words in the stimulus to recognized words in the lexicon based on various criteria such as word length and position in the stimulus. 
    Words that fit the criteria are matched to their respective slots in the stimulus. 
    """
    # Initialise a binary array to track recognized words in the lexicon
    new_recognized_words = np.zeros(lexicon_size)
    # Initialise a flag to track if a false guess has been made
    falseguess = False
    # Initialize counters for recognized nouns and verbs
    noun_count = 0
    ver_count = 0
    # Initialize an array to track part-of-speech recognition for each slot in the stimulus
    POSrecognition = ['' for _ in range(n_words_in_stim)]
    
    # Iterate over each slot in the stimulus to match words to slots
    for slot_to_check in range(0, n_words_in_stim):
        # Get the order in which the current slot should be checked
        slot_num = order_match_check[slot_to_check]

        # If the current slot has not been matched yet
        if len(stim_matched_slots[slot_num]) == 0:
            # Extract the word from the stimulus that corresponds to the current slot
            word_searched = stimulus.split()[slot_num]

            # Add the searched word to the list of all searched words
            all_words_searched.append(word_searched)
            print(f"Checking slot {slot_num} for word: {word_searched}")
            
            # Initialize a list to store intermediate results for debugging purposes
            intermediate_results = []
            for x in lexicon:
                # Check if the current word in the lexicon is an affix
                is_affix = x in affixes
                # Calculate the length of the current word in the lexicon
                word_len = len(x.replace('_', ''))
                # Check if the word length is similar to the length of the searched word
                is_similar_len = int(is_similar_word_length(pm, word_len, [len(word_searched)]))
                intermediate_results.append((x, is_affix, word_len, is_similar_len))

            # Generate a binary array indicating which words in the lexicon fit the length criteria and are not affixes
            recognWrdsFittingLen_np = above_thresh_lexicon_np * \
                        np.array([0 if x in affixes else int(is_similar_word_length(pm, len(x.replace('_', '')),
                            [len(word_searched)])) for x in lexicon])
            
            # If the current slot is the target slot, set the word frequency and predictability
            if sum(recognWrdsFittingLen_np):
                # Identify the word in the lexicon with the highest activity that fits the criteria
                highest = np.argmax(recognWrdsFittingLen_np * lexicon_word_activity_np)
                highest_word = lexicon[highest]
                print(f"word {highest_word.replace('_', '')} matched in slot {slot_num}, HIGHEST WORD")
                # Match the identified word to the current slot
                stim_matched_slots[slot_num] = highest_word
                new_recognized_words[highest] = 1
                # Reset the activity and threshold of the identified word
                above_thresh_lexicon_np[highest] = 0
                lexicon_word_activity_np[highest] = pm.min_activity

                # Check if the target word is in the stimulus
                if target in stimulus.split():
                    # If the current slot corresponds to the position of the target word in the stimulus
                    if stimulus.split().index(target) == slot_num:
                        # If the matched word is the target word, set the recognition flag to True
                        if target == highest_word.replace('_', ''):
                            recognized = True
                            print(f"matched word is target word")
                        else:
                            print(f"matched word is not target word, but should have been")
                            falseguess = True
                    else:
                        print(f"slot is not target slot")
                else:
                    # If the target word is not in the stimulus, it means the recognized word is a prime
                    print(f"prime recognized")


    return stim_matched_slots, new_recognized_words, above_thresh_lexicon_np, lexicon_word_activity_np, recognized, falseguess, POSrecognition, noun_count, ver_count, all_words_searched

def check_recognition_status(cur_cycle, pm, recognized, falseguess, lexicon_total_input_np, cycle_for_RT):
    """
    Checks the recognition status of the target and handles different scenarios based on the current cycle and recognition flags.

    Parameters:
    - cur_cycle (int): The current cycle or timestep.
    - pm (object): An object containing model parameters, including 'totalcycles' and 'trial_ends_on_key_press'.
    - recognized (bool): A flag indicating whether the target word has been recognized.
    - falseguess (bool): A flag indicating if a false guess has been made.
    - lexicon_total_input_np (numpy array): The total input for each word in the lexicon.

    Returns:
    - cycle_for_RT (int): The cycle at which the recognition time is determined.
    - end_trial (bool): A flag indicating whether to end the trial.
    """

    # Initialize end_trial flag to False
    end_trial = False

    # If target has not been recognized yet, update the cycle for RT
    if recognized == False:
        cycle_for_RT = cur_cycle

    # Check if the current cycle is the last one and print the recognition status
    if cur_cycle == pm.totalcycles - 1:
        print(f"last cycle reached! target recognized: {recognized}")

    # If the trial should end on a key press and the target has been recognized or a false guess has been made
    if pm.trial_ends_on_key_press and (recognized == True or falseguess == True):
        print(f"RECOGNIZED: {recognized}")
        print(f"FALSE GUESS: {falseguess}")

        # Check if all word activations are negative and print a warning if so
        check = any(lexicon_total_input_np > 0)
        if not check:
            print('WARNING: all word activations are negative. make sure inhibition/excitation balance in parameters is ok. You can set pm.plotting to True to see the inhibition values during the task')
        
        # Set end_trial flag to True
        end_trial = True

    return cycle_for_RT, end_trial

def determine_correctness(pm, grammatical, stim, trial, identif, recognized, target):
    """
    Determine the correctness of a trial based on task type and other conditions.

    Parameters:
    - pm: Model parameters.
    - grammatical: Boolean indicating if the stimulus is grammatical.
    - stim: Dictionary containing stimulus data.
    - trial: Current trial number.
    - identif: Boolean indicating identification status (used for 'Classification' task).
    - recognized: Boolean indicating if the word was recognized.
    - target: Target word for the trial.

    Returns:
    - correct: Boolean indicating if the trial was correct.
    - unrecognized_words: List containing the target word if it was not recognized.
    """

    correct = False
    unrecognized_words = []

    if pm.task_to_run == 'Transposed':
        if (grammatical and stim['condition'][trial] != 'normal') or \
           (not grammatical and stim['condition'][trial] == 'normal'):
            correct = False
        else:
            correct = True

    elif pm.task_to_run == 'Classification':
        correct = identif

    else:
        if not recognized:
            unrecognized_words.append(target)
            correct = False
        else:
            correct = True

    return correct, unrecognized_words

def compute_trial_metrics(cycle_for_RT, pm, target, word_thresh_dict, word_frequencies):
    """
    Compute reaction time and other trial metrics.

    Parameters:
    - cycle_for_RT: Cycle number for reaction time.
    - pm: Model parameters.
    - target: Target word for the trial.
    - word_thresh_dict: Dictionary mapping words to their thresholds.
    - word_frequencies: Dictionary mapping words to their frequencies.

    Returns:
    - reaction_time: Computed reaction time for the trial.
    - word_threshold: Threshold for the target word.
    - word_frequency: Frequency for the target word.
    """
    
    # Compute reaction time based on the cycle number and model parameters.
    reaction_time = ((cycle_for_RT + 1 - pm.blankscreen_cycles_begin) * pm.cycle_size) + 300
    print("reaction time: " + str(reaction_time) + " ms")
    
    # Retrieve word threshold and frequency for the target word.
    word_threshold = word_thresh_dict.get(target, "")
    word_frequency = word_frequencies.get(target, "")

    return reaction_time, word_threshold, word_frequency
