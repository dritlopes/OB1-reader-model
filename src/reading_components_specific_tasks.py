import numpy as np
from strsimpy.longest_common_subsequence import LongestCommonSubsequence
import nltk
import matplotlib.pyplot as plt
import os
import sys
import pickle
import warnings
import logging
logger = logging.getLogger(__name__)
from reading_helper_functions_specific_tasks import string_to_bigrams_and_locations

import torch
from torch import nn


lcs = LongestCommonSubsequence()

def extract_stem(word, prefixes, suffixes, affixes):
    """
    Identifies if the input word has an affix, and if so, it extracts the stem.
    Returns stem and matched affix
    """

    inferred_stem = None
    # Checks if affixes in the input are present in the input word.
    matching = [s for s in affixes if s in word]
    # if matched affix is same as word, then that word is the affix: skip it
    if any(matching) and matching[0] != word:
        if len(matching) > 1:  # if more than 1 affix recognized
            match = max(matching, key=len)  # take longest match (ity instead of y)
        else:
            match = matching[0]
        # Calculate the length of the stem by subtracting the length of the matched affix from the length of the input word
        stem_len = len(word.strip('_'))-len(match.strip('_'))
        #  If the matched affix is in the list of suffixes, extract the stem by removing the suffix
        if match in suffixes:
            inferred_stem = word.strip('_')[:stem_len]  # remove last part of word to get stem
            #print(f"Word: {word}, Suffix: {match}, Stem: {inferred_stem}")
        elif match in prefixes:
            # remove first part of word to get stem
            inferred_stem = word.strip('_')[stem_len:]
            #print(f"Word: {word}, Prefix: {match}, Stem: {inferred_stem}")
        else:
            raise NameError('affix not in suffixes nor in prefixes??')
    return inferred_stem, matching

def word_stem_match(simil_algo, max_edit_dist, short_word_cutoff, word, stem):
    """
    Checks if a passed word and stem match according to a specified similarity algorithms (either 'startswith', 'lcs', or 'lev')
        - startwith. check if the input word starts with the given stem (True, False)
        - Longest Common Subsequence (lcs). If the input word's length is greater than short_word_cutoff, check if the LCS distance between the word and stem is less than or equal to max_edit_dist (True, False)
             If the input word's length is less than or equal to short_word_cutoff, check if the LCS distance between the word and stem is exactly 0.
        - Levenshtein (lev).  If the input word's length is greater than short_word_cutoff, check if the Levenshtein distance between the word and stem is less than or equal to max_edit_dist.
              If the input word's length is less than or equal to short_word_cutoff, check if the Levenshtein distance between the word and stem is exactly 0.

    max_edit_dist - The maximum edit distance allowed for a match (applicable for 'lcs' and 'lev' algorithms).
    short_word_cutoff - A length threshold for short words, which are subject to stricter matching criteria.
    word - input word
    stem - The stem to be compared with the input word (extract_stem function)
    """

    if simil_algo == 'startswith':
        return word.startswith(stem)

    elif simil_algo == 'lcs':
        if len(word) > short_word_cutoff:  # if word is long, the distance must be above the threshold
            return lcs.distance(word, stem) <= max_edit_dist
        else:
            return lcs.distance(word, stem) == 0  # for short words, distance is stricter: must be 0

    elif simil_algo == 'lev':
        if len(word) > short_word_cutoff:
            return nltk.edit_distance(word, stem) <= max_edit_dist
        else:
            return nltk.edit_distance(word, stem) == 0

    else:
        raise NotImplementedError('this edit distance function is not implemented!')

def load_affixes(language):
    """
    Load prefix and suffix frequency dictionaries for the specified language.

    Parameters:
        language: The language of interest.
        
    Returns:
        A tuple containing two dictionaries: the prefix and suffix frequency dictionaries.

    Raises:
        ValueError: If an unsupported language is passed.
    """
    language_codes = {
        'english': 'en',
        'french': 'fr',
        'german': 'de'
    }
    language_code = language_codes.get(language)

    if not language_code:
        raise ValueError(f"Unsupported language in affix system: {language}")

    # Load prefix and suffix frequency dictionaries for the specified language.
    with open(f'data/suffix_frequency_{language_code}.dat', 'rb') as f:
        suffixes_dict = pickle.load(f)
    with open(f'data/prefix_frequency_{language_code}.dat', 'rb') as f:
        prefixes_dict = pickle.load(f)

    return prefixes_dict, suffixes_dict

def complex_stem_pairing(word, other_word, lexicon, lexicon_word_bigrams, lexicon_size, prefixes_dict, suffixes_dict, pm, word_overlap_matrix):
    """
    Checks for the presence of affixes in the given words and performs certain operations.

    Parameters:
        word: Index of the first word.
        other_word: Index of the second word.
        lexicon: The lexicon of words.
        affixes: A set of known affixes.
        prefixes: A set of known prefixes.
        suffixes: A set of known suffixes.
        pm: The parameter model, containing parameters like simil_algo, max_edit_dist, short_word_cutoff.
        complex_stem_pairs: A list of complex-stem pairs.
    """
    """
    Computes overlap counter for a pair of words.

    Parameters:
        word: Index of the first word.
        other_word: Index of the second word.
        lexicon_word_bigrams: Precomputed bigrams for each word in the lexicon.
        lexicon: The lexicon of words.
        n_known_words: Number of known words in the lexicon.
        pm: The parameter model, containing parameters like min_overlap.
    """
    lexicon_word_bigrams_set = {}
    bigrams_common = []
    bigrams_append = bigrams_common.append
    bigram_overlap_counter = 0
    for bigram in range(len(lexicon_word_bigrams[lexicon[word]])):
        if lexicon_word_bigrams[lexicon[word]][bigram] in lexicon_word_bigrams[lexicon[other_word]]:
            bigrams_append(lexicon_word_bigrams[lexicon[word]][bigram])
            lexicon_word_bigrams_set[lexicon[word]] = set(
                lexicon_word_bigrams[lexicon[word]])
            bigram_overlap_counter += 1

    monograms_common = []
    monograms_append = monograms_common.append
    monogram_overlap_counter = 0
    unique_word_letters = ''.join(set(lexicon[word]))

    for pos in range(len(unique_word_letters)):
        monogram = unique_word_letters[pos]
        if monogram in lexicon[other_word]:
            monograms_append(monogram)
            monogram_overlap_counter += 1
    #print(monograms_common)

    total_overlap_counter = bigram_overlap_counter + monogram_overlap_counter

    if word >= lexicon_size or other_word >= lexicon_size:
        total_overlap_counter = 0

    min_overlap = 2
    
    pickle_filename = f'data/complex_stem_pairs_{pm.task_to_run}.pkl'
    # Create lists of prefixes, suffixes, and affixes (combined prefixes and suffixes).
    suffixes = list(suffixes_dict.keys())
    prefixes = list(prefixes_dict.keys())
    affixes = prefixes + suffixes

    complex_stem_pairs = []
    # NV: affixes dont exert inhibition on normal words, and between each other
    if (lexicon[word] in affixes) or (lexicon[other_word] in affixes):
        affix_only = True  # marks whether one of 2 words is an affix, useful for later
        total_overlap_counter = 0
    else:
        affix_only = False

    # if word or other-word is not only an affix itself, and the 2 words aren't the same
    if not affix_only and lexicon[word] != lexicon[other_word]:
        # get stem from full word (for ex., weaken should output weak as inferred stem)
        inferred_stem_otherword, matching_otherword = extract_stem(lexicon[other_word], prefixes, suffixes, affixes)
        inferred_stem_word, matching_word = extract_stem(lexicon[word], prefixes, suffixes, affixes)

        # if word is affixed (matching contains affixes)
        # NV: determine if word-stem distance is within threshold, given max allowed edit distance, edit distance algorithm,
        # and cutoff (under cutoff (short words), stem and word must be exactly the same.)
        # here, we determined best values to be max_edit_dist = 1, cutoff=3, with algo = lcs.
        # cutoff 4 yields slightly better precision, for slightly worse recall.
        if (any(matching_otherword) and len(inferred_stem_otherword) > 1) and \
                word_stem_match(pm.simil_algo, pm.max_edit_dist, pm.short_word_cutoff,
                                lexicon[word].strip('_'), inferred_stem_otherword):
            complex_stem_pairs.append((lexicon[other_word], lexicon[word]))  # order:complex-stem (weaken, weak)
        elif (any(matching_word) and len(inferred_stem_word) > 1) and \
                word_stem_match(pm.simil_algo, pm.max_edit_dist, pm.short_word_cutoff,
                                lexicon[other_word].strip('_'), inferred_stem_word):
            complex_stem_pairs.append((lexicon[word], lexicon[other_word]))  # order:complex-stem (weaken, weak)

    if total_overlap_counter > min_overlap and word != other_word:
        # NV: remove min overlap from total?
        word_overlap_matrix[word,
                            other_word] = total_overlap_counter - min_overlap
        word_overlap_matrix[other_word,
                            word] = total_overlap_counter - min_overlap
    else:
        word_overlap_matrix[word, other_word] = 0
        word_overlap_matrix[other_word, word] = 0

    return total_overlap_counter, complex_stem_pairs, word_overlap_matrix


def check_if_complex_pair_is_related(prime, target, complex_stem_pairs):
    return (prime, target) in complex_stem_pairs or (target, prime) in complex_stem_pairs

def is_nonword(word, stim_data):
    # Get the condition of the word
    word_condition = stim_data[stim_data['stimulus'] == word]['condition']
    
    # Check if the word_condition Series is not empty
    if not word_condition.empty:
        # Compare the condition value to real-word conditions
        if word_condition.values[0] not in ['REALWORD/simple', 'REALWORD/complex']:
            return True
    return False

def get_nonwords_and_realwords(stim_data):
    """
    Returns a list of nonwords and a list of realwords from a stimulus dataframe.
    """
    nonwords = []
    realwords = []
    
    for _, row in stim_data.iterrows():
        word = row['stimulus']
        condition = row['condition']
        if condition not in ['REALWORD/simple', 'REALWORD/complex']:
            nonwords.append(word)
        else:
            realwords.append(word)
    
    return nonwords, realwords

def affix_modelling_underscores(word_freq_dict_temp):
    word_freq_dict = {}
    for word in word_freq_dict_temp.keys():
        word_freq_dict[f"_{word}_"] = word_freq_dict_temp[word]
    return word_freq_dict

def underscores_list(word_list):
    return [f"_{word}_" for word in word_list]

def lexicon_bigrams(lexicon):
    lexicon_word_bigrams = {}
    bigram_monogram_total = []
    
    for word in lexicon:
        word_local = word
        is_suffix = False
        is_prefix = False
        if word_local.startswith('_') and word_local.endswith('_'):
            word_local = word_local[1:-1]
        elif word_local.startswith('_'):
            word_local = word_local[1:]
            is_prefix = True
        elif word_local.endswith('_'):
            word_local = word_local[:-1]
            is_suffix = True
        else:
            raise SyntaxError("word does not start or stop with _ . Verify lexicon")

        # convert words into bigrams and their locations
        word_local = " "+word_local+" "
        (all_word_bigrams, bigramLocations) = string_to_bigrams_and_locations(word_local, is_prefix, is_suffix)
        # append to list of N ngrams
        lexicon_word_bigrams[word] = all_word_bigrams
        # bigrams and monograms total amount
        bigram_monogram_total.append(len(all_word_bigrams) + len(word.strip('_')))
    
    return lexicon_word_bigrams, bigram_monogram_total


def setup_inhibition_grid_specific_task(lexicon, LEXICON_SIZE, lexicon_word_bigrams, 
                          pm, n_known_words, affixes, prefixes, suffixes, individual_to_lexicon_indices, lexicon_word_bigrams_set):
    """
    This function sets up the inhibition grid.

    :param lexicon: The lexicon list of words.
    :param LEXICON_SIZE: Size of the lexicon.
    :param lexicon_word_bigrams: Bigrams associated with lexicon words.
    :param pm: Parameter module containing necessary parameters.
    :param n_known_words: Number of known words.
    :param logger: Logger for logging info.
    :param affixes: List of affixes.
    :param prefixes: List of prefixes.
    :param suffixes: List of suffixes.
    :param extract_stem: Function to extract stem from a word.
    :param word_stem_match: Function to match word stem.
    :param individual_to_lexicon_indices: Mapping of individual to lexicon indices.
    :return: word_overlap_matrix, word_inhibition_matrix
    """

    print("Setting up word-to-word inhibition grid...")
   

    # Set up the list of word inhibition pairs, with amount of bigram/monograms overlaps for every pair. Initialize inhibition matrix with false.
    # NV: COMMENT: here is actually built an overlap matrix rather than an inhibition matrix, containing how many bigrams of overlap any 2 words have
    word_inhibition_matrix = np.zeros((LEXICON_SIZE, LEXICON_SIZE), dtype=bool)
    word_overlap_matrix = np.zeros((LEXICON_SIZE, LEXICON_SIZE), dtype=int)

    complete_selective_word_inhibition = True  # NV: what does this do exactly? Move to parameters.py?
    overlap_list = {}

    # NV: first, try to fetch parameters of previous inhib matrix
    try:
        with open('Data/Inhib_matrix_params_latest_run.dat', "rb") as f:
            parameters_previous = pickle.load(f)

        size_of_file = os.path.getsize('Data/Inhibition_matrix_previous.dat')

        # NV: compare the previous params with the actual ones.
        # he idea is that the matrix is fully dependent on these parameters alone.
        # So, if the parameters are the same, the matrix should be the same.
        # The file size is also added as a check . Note: Could possibly be more elegant
        if str(lexicon_word_bigrams)+str(LEXICON_SIZE)+str(pm.min_overlap) +\
           str(complete_selective_word_inhibition)+str(n_known_words)+str(pm.affix_system) +\
           str(pm.simil_algo)+str(pm.max_edit_dist) + str(pm.short_word_cutoff)+str(size_of_file) \
           == parameters_previous:

            previous_matrix_usable = True  # FIXME: turn off if need to work on inihibition matrix specifically

        else:
            previous_matrix_usable = False
    except:
        
        previous_matrix_usable = False
    # print(f"word_inhibition_matrix: {word_overlap_matrix[0]}\n")
    # print(f"word_inhibition_matrix shape: {word_overlap_matrix.shape}\n")
    # NV: if the current parameters correspond exactly to the fetched params of the previous run, use that matrix
    if previous_matrix_usable:
        with open('Data/Inhibition_matrix_previous.dat', "rb") as f:
            word_overlap_matrix = pickle.load(f)
        print('using pickled inhibition matrix')
        
    # NV: else, build it
    else:
        print('building inhibition matrix')
        
        # print(f"word_inhibition_matrix: {word_overlap_matrix}\n")
        # print(f"word_inhibition_matrix shape: {word_overlap_matrix.shape}\n")

        #overlap_percentage_matrix = np.zeros((LEXICON_SIZE, LEXICON_SIZE))
        complex_stem_pairs = []
        overlap_more_than_zero = []
        overlap_more_than_one = []

        for other_word in range(LEXICON_SIZE):

            # as loop is symmetric, only go through every pair (word1-word2 or word2-word1) once.
            for word in range(other_word, LEXICON_SIZE):
                

                ### NV: 1. calculate monogram and bigram overlap
                
                # NV: bypass to investigate the effects of word-length-independent inhibition
                # if not is_similar_word_length(lexicon[word], lexicon[other_word]) or lexicon[word] == lexicon[other_word]: # Take word length into account here (instead of below, where act of lexicon words is determined)
                bigrams_common = []
                bigrams_append = bigrams_common.append
                bigram_overlap_counter = 0
                for bigram in range(len(lexicon_word_bigrams[lexicon[word]])):
                    if lexicon_word_bigrams[lexicon[word]][bigram] in lexicon_word_bigrams[lexicon[other_word]]:
                        bigrams_append(lexicon_word_bigrams[lexicon[word]][bigram])
                        lexicon_word_bigrams_set[lexicon[word]] = set(
                            lexicon_word_bigrams[lexicon[word]])
                        bigram_overlap_counter += 1

                monograms_common = []
                monograms_append = monograms_common.append
                monogram_overlap_counter = 0
                unique_word_letters = ''.join(set(lexicon[word]))

                for pos in range(len(unique_word_letters)):
                    monogram = unique_word_letters[pos]
                    if monogram in lexicon[other_word]:
                        monograms_append(monogram)
                        monogram_overlap_counter += 1

                # take into account both bigrams and monograms for inhibition counters (equally)
                total_overlap_counter = bigram_overlap_counter + monogram_overlap_counter

                # if word or other word is larger than the initial lexicon
                # (without PSC), overlap counter = 0, because words that are not
                # known should not inhibit
                if word >= n_known_words or other_word >= n_known_words:
                    total_overlap_counter = 0
                min_overlap = pm.min_overlap  # MM: currently 2
                ### NV: 2. take care of affix system, if relevant
                
                if pm.affix_system:

                    # NV: affixes dont exert inhibition on normal words, and between each other
                    if (lexicon[word] in affixes) or (lexicon[other_word] in affixes):
                        affix_only = True  # marks whether one of 2 words is an affix, useful for later
                        total_overlap_counter = 0
                    else:
                        affix_only = False

                    # if word or other-word is not only an affix itself, and the 2 words arent the same
                    if not(affix_only) and lexicon[word] != lexicon[other_word]:

                        # get stem from full word (for ex., weaken should output weak as inferred stem)
                        inferred_stem_otherword, matching_otherword = extract_stem(
                            lexicon[other_word], prefixes, suffixes, affixes)
                        inferred_stem_word, matching_word = extract_stem(
                            lexicon[word], prefixes, suffixes, affixes)

                        # if word is affixed (matching contains affixes)
                        # NV: determine if word-stem distance is within threshold, given max allowed edit distance, edit distance algorithm,
                        # and cutoff (under cutoff (short words), stem and word must be exactly the same.)
                        # here, we determined best values to be max_edit_dist = 1, cutoff=3, with algo = lcs.
                        # cutoff 4 yields slightly better precision, for slightly worse recall.
                        if (any(matching_otherword) and len(inferred_stem_otherword) > 1) and \
                            word_stem_match(pm.simil_algo, pm.max_edit_dist, pm.short_word_cutoff,
                                            lexicon[word].strip('_'), inferred_stem_otherword):
                            complex_stem_pairs.append(
                                (lexicon[other_word], lexicon[word]))  # order:complex-stem (weaken, weak)

                        elif (any(matching_word) and len(inferred_stem_word) > 1) and \
                            word_stem_match(pm.simil_algo, pm.max_edit_dist, pm.short_word_cutoff,
                                            lexicon[other_word].strip('_'), inferred_stem_word):
                            complex_stem_pairs.append(
                                (lexicon[word], lexicon[other_word]))  # order:complex-stem (weaken, weak)
                ### Set inhibition values
                
                if complete_selective_word_inhibition:
                    if total_overlap_counter > min_overlap and word != other_word:

                        word_overlap_matrix[word,
                                            other_word] = total_overlap_counter - min_overlap
                        word_overlap_matrix[other_word,
                                            word] = total_overlap_counter - min_overlap
                        if total_overlap_counter - min_overlap > 0:
                            overlap_more_than_zero.append((lexicon[word], lexicon[other_word]))
                        if total_overlap_counter - min_overlap > 1:
                            overlap_more_than_one.append((lexicon[word], lexicon[other_word]))
                    else:
                        word_overlap_matrix[word, other_word] = 0
                        word_overlap_matrix[other_word, word] = 0
                        
                else:  # is_similar_word_length
                    if total_overlap_counter > min_overlap:
                        word_inhibition_matrix[word, other_word] = True
                        word_inhibition_matrix[other_word, word] = True
                        overlap_list[word, other_word] = total_overlap_counter - min_overlap
                        overlap_list[other_word, word] = total_overlap_counter - min_overlap
                        sys.exit(
                            'Make sure to use slow version, fast/vectorized version not compatible')

    
        # for word1, word2 in complex_stem_pairs:
        #     idx1, idx2 = lexicon.index(word1), lexicon.index(word2)
        #     word_inhibition_matrix[idx1, idx2] = 0
        #     word_inhibition_matrix[idx2, idx1] = 0
        #     inhibition_value = word_inhibition_matrix[idx1, idx2]
        for word1, word2 in complex_stem_pairs:
            word_overlap_matrix[lexicon.index(word1), lexicon.index(word2)] = 0
            word_overlap_matrix[lexicon.index(word2), lexicon.index(word1)] = 0
            

        # Save overlap matrix, with individual words selected (why is this needed?)
        output_inhibition_matrix = 'Data/Inhibition_matrix_'+pm.language+'.dat'
        with open(output_inhibition_matrix, "wb") as f:
            pickle.dump(np.sum(word_overlap_matrix, axis=0)[individual_to_lexicon_indices], f)
    print("Inhibition grid ready.")

    return word_overlap_matrix, word_inhibition_matrix, overlap_more_than_zero, overlap_more_than_one

def semantic_processing(text, tokenizer, language_model, prediction_flag, top_k = 10, threshold = None, device = None):

    pred_info = dict()
    #print(text, len(text))
    for i in range(1, len(text)):

        sequence = ' '.join(text[:i])
        # pre-process text
        encoded_input = tokenizer(sequence, return_tensors='pt')
        if device:
            encoded_input.to(device)
        # output contains at minimum the prediction scores of the language modelling head,
        # i.e. scores for each vocab token given by a feed-forward neural network
        output = language_model(**encoded_input)
        # logits are prediction scores of language modelling head;
        # of shape (batch_size, sequence_length, config.vocab_size)
        logits = output.logits[:, -1, :]
        # convert raw scores into probabilities (between 0 and 1)
        probabilities = nn.functional.softmax(logits, dim=1)

        # # add target word, also if subtoken
        target_word = text[i]
        if prediction_flag == 'gpt2':
            target_word = ' ' + text[i]
        target_token = tokenizer.encode(target_word, return_tensors='pt')

        if top_k == 'target_word':
            if target_token.size(dim=1) > 0:
                top_tokens = [target_word]
                target_id = target_token[0][0]
                # deals with quirk from llama of having <unk> as first token
                if prediction_flag == 'llama':
                    decoded_token = [tokenizer.decode(token) for token in target_token[0]]
                    if decoded_token[0] == '<unk>':
                        target_id = target_token[0][1]
                top_probabilities = [float(probabilities[0,target_id])]
                pred_info[i] = (top_tokens, top_probabilities)
        else:
            k = top_k
            if top_k == 'all':
                # top k is the number of probabilities above threshold
                if threshold:
                    above_threshold = torch.where(probabilities > threshold, True, False)
                    only_above_thrs = torch.masked_select(probabilities, above_threshold)
                    k = len(only_above_thrs)
                else:
                    k = len(probabilities[0])
            top_tokens = [tokenizer.decode(id.item()) for id in torch.topk(probabilities, k=k)[1][0]]
            top_probabilities = [float(pred) for pred in torch.topk(probabilities, k=k)[0][0]]
            # add target word if among top pred, also if subtoken
            target_tokens = [tokenizer.decode(token) for token in target_token[0]]
            target_tokens = [token for token in target_tokens if token != '<unk>']
            if target_tokens[0] in top_tokens:
                loc = top_tokens.index(target_tokens[0])
                top_tokens[loc] = target_word
            pred_info[i] = (top_tokens, top_probabilities)

    return pred_info