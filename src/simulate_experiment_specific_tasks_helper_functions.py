from matplotlib import pyplot as plt
import numpy as np
import math

from reading_components_specific_tasks import affix_modelling_underscores, get_nonwords_and_realwords, setup_inhibition_grid_specific_task, underscores_list
from reading_helper_functions_specific_tasks import get_threshold
from utils_specific_tasks import combine_affix_frequency_files, get_suffix_file, get_word_freq, process_words_underscores, save_lexicon, stimuli_word_threshold_dict

def process_and_filter_words(pm):
    """
    Process and filter non-words and real-words based on the provided parameters.

    Parameters:
    - pm: The parameter object containing settings and stimuli.

    Returns:
    - tokens: A list of processed words without duplicates.
    """
    realwords = []
    nonwords = []
    nonwords_no_underscore = []
    # Handle non-words and real-words if necessary
    if pm.are_nonwords:
        # Splitting nonwords and real words using our helper function.
        nonwords_no_underscore, realwords_no_underscore = get_nonwords_and_realwords(pm.stim)
        # Giving words the underscores
        realwords, realwords_lengths = process_words_underscores(realwords_no_underscore)
        nonwords, nonwords_lengths = process_words_underscores(nonwords_no_underscore)   
    
    # If the stimuli is just a string, split it. Otherwise, process each bit.
    if type(pm.stim_all) == str:
        tokens = pm.stim_all.split(' ')
    else:
        # Create a flattened list of words from all stimuli.
        tokens = [token for stimulus in pm.stim_all for token in stimulus.split(' ')]

    # Remove duplicates while keeping the order
    seen = set()
    tokens_no_duplicates = [token for token in tokens if not (token in seen or seen.add(token))]

    return tokens_no_duplicates, nonwords_no_underscore, realwords, nonwords

def process_tokens(pm, tokens, nonwords_no_underscore=None):
    """
    Process tokens based on the provided parameters.
    
    Parameters:
    - pm: The parameter object containing settings.
    - tokens: A list of words/tokens to be processed.
    - nonwords_no_underscore: A list of nonwords without underscores.

    Returns:
    - tokens: A list of processed words.
    - word_frequencies: A dictionary of word frequencies.
    """
    
    # If there's a priming task and not dealing with nonwords, add the primes.
    if pm.is_priming_task and not pm.are_nonwords:
        tokens.extend([token for stimulus in list(pm.stim["prime"]) for token in stimulus.split(' ')])
    
    # Filter out nonwords from tokens
    if pm.are_nonwords and nonwords_no_underscore:
        tokens = [token for token in tokens if token not in nonwords_no_underscore]

    # If there's an affix system, process tokens using underscores_list function
    if pm.affix_system:
        tokens = underscores_list(tokens)

    # Compute word frequencies
    word_frequencies = get_word_freq(pm, set([token.lower() for token in tokens]))

    return tokens, word_frequencies

def process_affixes(pm, word_frequencies):
    """
    Process word frequencies based on the affix system provided in parameters.
    
    Parameters:
    - pm: The parameter object containing settings.
    - word_frequencies: A dictionary of word frequencies.

    Returns:
    - word_frequencies: A processed dictionary of word frequencies.
    - affixes (optional): A list of affixes (both prefixes and suffixes).
    """
    
    # Modify word frequencies if there's an affix system
    if pm.affix_system:
        word_frequencies = affix_modelling_underscores(word_frequencies)
    
    # Process suffixes if there's an affix system
    if pm.affix_system:
        # Check for the combined affix system
        combine_affix_frequency_files(pm.language)
        
        # Process the suffixes
        suffix_freq_dict_temp = get_suffix_file(pm.language)
        suffix_freq_dict = {}
        prefixes = []
        prefix_freq_dict = {}
        for word in suffix_freq_dict_temp.keys():
            suffix_freq_dict[f"{word}_"] = suffix_freq_dict_temp[word]
        suffixes = list(suffix_freq_dict.keys())
        affixes = prefixes + suffixes

        # Merge suffixes with the word frequencies
        word_frequencies = word_frequencies | suffix_freq_dict

    # Return word frequencies and affixes if affix system is in place
    if pm.affix_system:
        return word_frequencies, affixes, prefixes, suffixes
    else:
        return word_frequencies
    
def build_lexicon(tokens, word_frequencies, pm):
    """
    Constructs the lexicon from given tokens and word frequencies.

    Args:
    - tokens (list): List of word tokens.
    - word_frequencies (dict): Dictionary of word frequencies.
    - pm (object): Parameter object containing various simulation parameters.

    Returns:
    - lexicon (list): Combined list of unique words.
    - lexicon_exp (list): Expanded lexicon containing unique words without repetition.
    - lexicon_normalized_word_inhibition (float): Computed normalized word inhibition.
    """
    
    # Combine tokens and word frequencies to form the lexicon.
    max_frequency = max(word_frequencies.values())
    lexicon = list(set(tokens) | set(word_frequencies.keys()))
    lexicon_size = len(lexicon)

    # Construct the expanded lexicon.
    lexicon_exp = []
    for word in tokens:  
        if word not in lexicon_exp:
            lexicon_exp.append(word)
    if len(word_frequencies) > 0:
        for freq_word in word_frequencies.keys():
            if freq_word.lower() not in lexicon_exp:
                lexicon_exp.append(freq_word.lower())
    lexicon_exp_size = len(lexicon_exp)
    lexicon_word_bigrams_set = {}

    # Compute normalized word inhibition.
    lexicon_normalized_word_inhibition = (100.0 / lexicon_size) * pm.word_inhibition
    print(f"LEXICON NORMALIZED WORD INHIBITION: {lexicon_normalized_word_inhibition}\n")

    # Save lexicon for consulting purposes.
    save_lexicon(lexicon, pm.task_to_run)

    return lexicon, lexicon_exp, lexicon_normalized_word_inhibition, max_frequency, lexicon_size, lexicon_exp_size, lexicon_word_bigrams_set

def identify_realwords(pm, realwords, tokens):
    """
    Determine the word set based on whether nonwords are present.

    Args:
    - pm: containing various simulation parameters.
    - realwords (list): List of real words.
    - tokens (list): List of word tokens.

    Returns:
    - list: Words from stimuli based on the presence of nonwords.
    """
    
    if pm.are_nonwords:
        return realwords
    else:
        return tokens
    
def compute_word_threshold(words_from_stimuli, word_frequencies, max_frequency, wordfreq_p, max_threshold):
    """
    Compute the word threshold based on word frequencies.

    Args:
    - words_from_stimuli (list): Words extracted from the stimuli.
    - word_frequencies (dict): A dictionary containing word frequencies.
    - max_frequency (int): Maximum frequency value.
    - wordfreq_p (float): Word frequency parameter.
    - max_threshold (int): Maximum threshold value.

    Returns:
    - dict: Dictionary containing word thresholds.
    """
    
    value_list = np.sort(list(word_frequencies.values()))
    value_to_insert = value_list[7]
    
    word_thresh_dict = stimuli_word_threshold_dict(words_from_stimuli, word_frequencies, max_frequency, wordfreq_p, max_threshold, value_to_insert)
    
    return word_thresh_dict

def generate_lexicon_mappings(tokens, lexicon, word_frequencies, max_frequency, wordfreq_p, max_threshold):
    """
    Generate various lexicon mappings and arrays.

    Args:
    - tokens (list): List of individual words.
    - lexicon (list): List of words in the lexicon.
    - word_frequencies (dict): Dictionary containing word frequencies.
    - max_frequency (int): Maximum frequency value.
    - wordfreq_p (float): Word frequency parameter.
    - max_threshold (int): Maximum threshold value.

    Returns:
    - tuple: Contains individual_to_lexicon_indices, lexicon_thresholds_np, lexicon_index_dict, lexicon_word_activity.
    """
    
    # lexicon indices for each word of text (individual_words)
    individual_to_lexicon_indices = np.zeros((len(tokens)), dtype=int)
    for i, word in enumerate(tokens):
        individual_to_lexicon_indices[i] = lexicon.index(word)

    lexicon_thresholds_np = np.zeros(len(lexicon))
    lexicon_index_dict = {}
    lexicon_word_activity = {}
    
    for i, word in enumerate(lexicon):
        lexicon_thresholds_np[i] = get_threshold(word, word_frequencies, max_frequency,  wordfreq_p, max_threshold)
        lexicon_index_dict[word] = i
        lexicon_word_activity[word] = 0.0

    return individual_to_lexicon_indices, lexicon_thresholds_np, lexicon_index_dict, lexicon_word_activity


def compute_overlap_and_inhibition_matrices(lexicon, lexicon_size, lexicon_word_bigrams, pm, affixes, prefixes, suffixes, individual_to_lexicon_indices, lexicon_word_bigrams_set):
    """
    
    Computes the word overlap and inhibition matrices and prints relevant information.

    Args:
        lexicon (list): List of words in the lexicon.
        lexicon_size (int): Size of the lexicon.
        lexicon_word_bigrams (dict): Dictionary containing bigrams for each word in the lexicon.
        pm (object): An object that contains parameter settings.
        affixes (list): List of affixes.
        prefixes (list): List of prefixes.
        suffixes (list): List of suffixes.
        individual_to_lexicon_indices (np.array): Array mapping individual words to their lexicon indices.
        lexicon_word_bigrams_set (dict): Dictionary with lexicon words as keys and their bigrams as values.

    """
    # Complex stem pairing is done in setup_inhibition_grid
    word_overlap_matrix, word_inhibition_matrix, overlap_more_than_zero, overlap_more_than_one = setup_inhibition_grid_specific_task(lexicon, lexicon_size, lexicon_word_bigrams, 
                          pm, lexicon_size, affixes, prefixes, suffixes, individual_to_lexicon_indices, lexicon_word_bigrams_set)
    
    print(f"Type of word_overlap_matrix: {type(word_overlap_matrix)}")
    print(f"Shape of word_overlap_matrix: {word_overlap_matrix.shape if hasattr(word_overlap_matrix, 'shape') else 'Not available'}")

    print(f"Type of word_inhibition_matrix: {type(word_inhibition_matrix)}")
    print(f"Shape of word_inhibition_matrix: {word_inhibition_matrix.shape if hasattr(word_inhibition_matrix, 'shape') else 'Not available'}")

    print("Subset of word_overlap_matrix:")
    print(word_overlap_matrix[:5, :5] if hasattr(word_overlap_matrix, 'shape') else word_overlap_matrix)

    print("Subset of word_inhibition_matrix:")
    print(word_inhibition_matrix[:5, :5] if hasattr(word_inhibition_matrix, 'shape') else word_inhibition_matrix)

    if hasattr(word_overlap_matrix, 'isnan'):
        print(f"Number of NaN values in word_overlap_matrix: {np.isnan(word_overlap_matrix).sum()}")
    if hasattr(word_inhibition_matrix, 'isnan'):
        print(f"Number of NaN values in word_inhibition_matrix: {np.isnan(word_inhibition_matrix).sum()}")

    print(f"Overlap more than zero: {len(overlap_more_than_zero)}\n")
    print(f"Overlap more than one: {len(overlap_more_than_one)}\n")
    
    return word_overlap_matrix, word_inhibition_matrix, overlap_more_than_zero, overlap_more_than_one

def display_overlap_and_inhibition_matrices(word_inhibition_matrix, word_overlap_matrix, pm):
    """
    Display the Word Inhibition and Word Overlap matrices.

    Args:
        word_inhibition_matrix (np.array): The word inhibition matrix.
        word_overlap_matrix (np.array): The word overlap matrix.
        pm (object): An object that contains parameter settings.
    """
    if not pm.display_overlap_inhib_matrix:
        return

    # Display Word Inhibition Matrix
    plt.imshow(word_inhibition_matrix > 0, cmap='gray_r')
    plt.title("Word Inhibition Matrix")
    plt.colorbar(ticks=[0, 1], label='Overlap')
    plt.show()

    # Display Word Overlap Matrix
    plt.imshow(word_overlap_matrix > 0, cmap='gray_r')
    plt.title("Word Overlap Matrix")
    plt.colorbar(ticks=[0, 1], label='Overlap')
    plt.show()