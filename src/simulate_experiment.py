import logging
import numpy as np
import sys
import pandas as pd
import pickle
import os

from utils import get_stimulus_text_from_file, get_word_freq, get_pred_values, check_previous_inhibition_matrix


logger = logging.getLogger(__name__)


def getStimulusSpacePositions(stimulus):  # NV: get index of spaces in stimulus

    stimulus_space_positions = []
    for letter_position in range(len(stimulus)):
        if stimulus[letter_position] == " ":
            stimulus_space_positions.append(letter_position)

    return stimulus_space_positions


def getNgramEdgePositionWeight(ngram, ngramLocations, stimulus_space_locations):

    ngramEdgePositionWeight = 0.5  # This is just a default weight; in many cases, it's changed
    # to 1 or 2, as can be seen below.
    max_weight = 2.

    if len(ngram) == 2:
        first = ngramLocations[0]
        second = ngramLocations[1]

        if (first-1) in stimulus_space_locations and (second+1) in stimulus_space_locations:
            ngramEdgePositionWeight = max_weight
        elif (first+1) in stimulus_space_locations and (second-1) in stimulus_space_locations:
            ngramEdgePositionWeight = max_weight
        elif (first-1) in stimulus_space_locations or (second+1) in stimulus_space_locations:
            ngramEdgePositionWeight = 1.
        elif (first+1) in stimulus_space_locations or (second-1) in stimulus_space_locations:
            ngramEdgePositionWeight = 1.
    else:
        letter_location = ngramLocations
        # One letter word
        if letter_location-1 in stimulus_space_locations and letter_location+1 in stimulus_space_locations:
            ngramEdgePositionWeight = max_weight
        # letter at the edge
        elif letter_location-1 in stimulus_space_locations or letter_location+1 in stimulus_space_locations:
            ngramEdgePositionWeight = 1.

    return ngramEdgePositionWeight


def stringToBigramsAndLocations(stimulus,pm):

    """Returns list with all unique open bigrams that can be made of the stim, and their respective locations
    (called 'word' for historic reasons), restricted by the maximum gap between two intervening letters."""

    # For the current stimulus, bigrams will be made. Bigrams are only made
    # for letters that are within a range of 4 from each other; (gap=3)
    # Bigrams that contain word boundary letters have more weight.
    # This is done by means of locating spaces in stimulus, and marking
    # letters around space locations (as well as spaces themselves), as
    # indicators of more bigram weight.

    stimulus_space_positions = getStimulusSpacePositions(stimulus)
    stimulus = "_"+stimulus+"_"

    allBigrams = []
    bigramsToLocations = {}
    gap = pm.bigram_gap  # None = no limit
    if gap == None:
        for first in range(len(stimulus) - 1):
            if(stimulus[first] == " "):
                continue
            for second in range(first + 1, len(stimulus)):
                if(stimulus[second] == " "):
                    break
                bigram = stimulus[first]+stimulus[second]
                if bigram != '  ':
                    if not bigram in allBigrams:
                        allBigrams.append(bigram)
                    bigramEdgePositionWeight = getNgramEdgePositionWeight(
                        bigram, (first, second), stimulus_space_positions)
                    if(bigram in bigramsToLocations.keys()):
                        bigramsToLocations[bigram].append((first, second, bigramEdgePositionWeight))
                    else:
                        bigramsToLocations[bigram] = [(first, second, bigramEdgePositionWeight)]
    else:
        for first in range(len(stimulus) - 1):
            # NV: this code implant is meant to insert special suffix bigrams in bigram list
            if(stimulus[first] == " "):  # NV: means that first letter is index 1 or last+1
                if first == 1:  # NV: if it is in the beginning of word
                    second_alt = 2  # NV: _alt not to interfere with loop variables
                    bigram = '_'+stimulus[second_alt]
                    if not bigram in allBigrams:
                        allBigrams.append(bigram)
                    bigramEdgePositionWeight = getNgramEdgePositionWeight(bigram, (first, second_alt), stimulus_space_positions)
                    if(bigram in bigramsToLocations.keys()):
                        bigramsToLocations[bigram].append((first, second_alt, bigramEdgePositionWeight))
                    else:
                        bigramsToLocations[bigram] = [(first, second_alt, bigramEdgePositionWeight)]
                    continue
                elif first == len(stimulus)-2:  # NV: if first letter is the end space
                    first_alt = -3  # NV: index of last letter
                    # NV: get the actual index (you do +, because first alt is a negative number)
                    first_alt = len(stimulus)+first_alt
                    second_alt = -2  # NV: index of space after last letter
                    second_alt = len(stimulus)+second_alt
                    bigram = stimulus[first_alt]+'_'
                    if not bigram in allBigrams:
                        allBigrams.append(bigram)
                    bigramEdgePositionWeight = getNgramEdgePositionWeight(bigram, (first_alt, second_alt), stimulus_space_positions)
                    if(bigram in bigramsToLocations.keys()):
                        bigramsToLocations[bigram].append((first_alt, second_alt, bigramEdgePositionWeight))
                    else:
                        bigramsToLocations[bigram] = [(first_alt, second_alt, bigramEdgePositionWeight)]
                    continue

            # NV:pick letter between first+1 and first+1+gap+1 (end of bigram max length), as long as that is smaller than end of word
            for second in range(first + 1, min(first+1+gap+1, len(stimulus))):
                if(stimulus[second] == " "):  # NV: if that is second lettter, you know you have reached the end of possible bigrams
                    # NV: break out of second loop if second stim is __. This means symbols before word, or when end of word is reached.
                    break
                bigram = stimulus[first]+stimulus[second]
                if bigram != '  ':
                    if not bigram in allBigrams:
                        allBigrams.append(bigram)
                    bigramEdgePositionWeight = getNgramEdgePositionWeight(bigram, (first, second), stimulus_space_positions)
                    if(bigram in bigramsToLocations.keys()):
                        bigramsToLocations[bigram].append((first, second, bigramEdgePositionWeight))
                    else:
                        bigramsToLocations[bigram] = [(first, second, bigramEdgePositionWeight)]

    return allBigrams, bigramsToLocations


def get_threshold(word, word_freq_dict, max_frequency, freq_p, max_threshold):  # word_pred_dict,pred_p

    # should always ensure that the maximum possible value of the threshold doesn't exceed the maximum allowable word activity
    # let threshold be fun of word freq. freq_p weighs how strongly freq is (1=max, then thresh. 0 for most freq. word; <1 means less havy weighting)
    # from 0-1, inverse of frequency, scaled to 0(highest freq)-1(lowest freq)
    word_threshold = max_threshold
    try:
        word_frequency = word_freq_dict[word]
        # threshold values between 0.8 and 1
        word_threshold = max_threshold * \
            ((max_frequency/freq_p) - word_frequency) / (max_frequency/freq_p)
    except KeyError:
        pass
    # GS Only lower threshold for short words
    # if len(word) < 4:
    #    word_threshold = word_threshold/3
    # return (word_frequency_multiplier * word_predictability_multiplier) * (pm.start_nonlin - (pm.nonlin_scaler*(math.exp(pm.wordlen_nonlin*len(word)))))

    return word_threshold # /1.4)

def is_similar_word_length(len1, len2, len_sim_constant):

    is_similar = False
    # NV: difference of word length  must be within 15% of the length of the longest word
    if abs(len1-len2) < (len_sim_constant * max(len1, len2)):
        is_similar = True

    return is_similar


def compute_bigram_overlap():
    pass


def build_word_inhibition_matrix(lexicon,lexicon_word_bigrams,pm,tokens_to_lexicon_indices):

    complete_selective_word_inhibition = True
    lexicon_size = len(lexicon)
    word_overlap_matrix = np.zeros((lexicon_size, lexicon_size), dtype=int)
    word_inhibition_matrix = np.zeros((lexicon_size, lexicon_size), dtype=bool)

    for word_1_index in range(lexicon_size):
        for word_2_index in range(word_1_index+1,lexicon_size): # make sure word1-word2, but not word2-word1 or word1-word1.
            word1, word2 = lexicon[word_1_index], lexicon[word_2_index]
            if not is_similar_word_length(len(word1), len(word2), pm.word_length_similarity_constant):
                continue
            else:
                bigram_common = list(set(lexicon_word_bigrams[word1]) & set(lexicon_word_bigrams[word2]))
                n_bigram_overlap = len(bigram_common)
                monograms_common = list(set(word1) & set(word2))
                n_monogram_overlap = len(monograms_common)
                n_total_overlap = n_bigram_overlap + n_monogram_overlap

                if n_total_overlap > pm.min_overlap:
                    word_overlap_matrix[word1, word2] = n_total_overlap - pm.min_overlap
                    if not complete_selective_word_inhibition:
                        word_overlap_matrix[word2, word1] = n_total_overlap - pm.min_overlap
                        word_inhibition_matrix[word1, word2] = True
                        word_inhibition_matrix[word2, word1] = True
                else:
                    word_overlap_matrix[word1, word2] = 0

    output_inhibition_matrix = 'Data/Inhibition_matrix_' + pm.short[pm.language] + '.dat'
    with open(output_inhibition_matrix, "wb") as f:
        pickle.dump(np.sum(word_overlap_matrix, axis=0)[tokens_to_lexicon_indices], f) # TODO check this

    with open('../data/Inhib_matrix_params_latest_run.dat', "wb") as f:
        pickle.dump(str(lexicon_word_bigrams) + str(lexicon_size) + str(pm.min_overlap) +
                    str(complete_selective_word_inhibition) + # str(n_known_words) #str(pm.affix_system) +
                    str(pm.simil_algo) + str(pm.max_edit_dist) + str(pm.short_word_cutoff) + str(size_of_file), f)

    return word_overlap_matrix


def simulate_experiment(pm, outfile_results, outfile_unrecognized):

    # TODO adapt code to add affix system in a flexible manner!
    # TODO adapt code to add grammar in a flexible manner!

    text = list(pm.stim.str.split(' ',expand=True).stack().unique())

    if pm.is_priming_task:
        text.extend(list(pm.stim['prime'].str.split(
            ' ', expand=True).stack().unique()))

    tokens = [word.strip() for word in text if word.strip() != '']
    lengths = [len(token) for token in tokens]

    if pm.task_to_run == 'EmbeddedWords_German':
        tokens = [f"_{token.lower()}_" for token in tokens]

    cleaned_words = [token.replace(".", "").lower() for token in set(tokens)]
    word_frequencies = get_word_freq(pm,cleaned_words)
    pred_values = get_pred_values(pm,cleaned_words)

    max_frequency_key = max(word_frequencies, key=word_frequencies.get)
    max_frequency = word_frequencies[max_frequency_key]

    total_n_words = len(tokens)
    lexicon = list(set(tokens))
    lexicon_size = len(lexicon) # TODO check why lexicon same as tokens in input

    lexicon_normalized_word_inhibition = (100.0 / lexicon_size) * pm.word_inhibition

    # array with recognition flag for each word position in the text is set to true when a word whose length is similar to that of the fixated word, is recognised so if it fulfills the condition is_similar_word_length(fixated_word,other_word)
    recognized_position_flag = np.zeros(total_n_words, dtype=bool)
    # array with recognition flag for each word in the text, it is set to true whenever the exact word from the stimuli is recognized
    recognized_word_at_position_flag = np.zeros(total_n_words, dtype=bool)
    recognized_word_at_cycle = np.empty(total_n_words, dtype=int)
    recognized_word_at_cycle.fill(-1)
    # array which stores the history of regressions, is set to true at a certain position in the text when a regression is performed to that word
    regression_flag = np.zeros(total_n_words, dtype=bool)

    # TODO check which variables are used where, replace their location for better readability
    # # Set activation of all words in lexicon to zero and make bigrams for each word.
    # lexicon_word_activity = {}
    # lexicon_word_bigrams_set = {}
    # lexicon_index_dict = {}
    #
    # # Lexicon word measures
    # lexicon_word_activity_np = np.zeros((lexicon_size), dtype=float)
    # lexicon_word_inhibition_np = np.zeros((lexicon_size), dtype=float)
    # lexicon_word_inhibition_np2 = np.zeros((lexicon_size), dtype=float)
    # lexicon_active_words_np = np.zeros((lexicon_size), dtype=int)
    # word_input_np = np.zeros((lexicon_size), dtype=float)
    # lexicon_thresholds_np = np.zeros((lexicon_size), dtype=float)

    # define word activation thresholds
    word_thresh_dict = {}
    if pm.frequency_flag:
        for word in lexicon:
            word_thresh_dict[word] = get_threshold(word,word_frequencies,max_frequency,pm.wordfreq_p,pm.max_threshold)

    # lexicon indices for each word of text
    tokens_to_lexicon_indices = np.zeros((total_n_words), dtype=int)
    for i, word in enumerate(tokens):
        tokens_to_lexicon_indices[i] = lexicon.index(word)

    # lexicon bigram dict
    lexicon_word_bigrams = {}
    n_ngrams_lexicon = []  # GS list with amount of ngrams per word in lexicon # TODO maybe delete this, unnecessary variable, too many to keep track already
    for word in lexicon:
        word_local = " " + word + " "
        all_word_bigrams, bigramLocations = stringToBigramsAndLocations(word_local,pm)
        lexicon_word_bigrams[word] = all_word_bigrams
        n_ngrams_lexicon.append(len(all_word_bigrams) + len(word))

    # set up word-to-word inhibition matrix
    previous_matrix_usable = check_previous_inhibition_matrix(lexicon,pm,lexicon_word_bigrams)
    if previous_matrix_usable:
        with open('../data/Inhibition_matrix_previous.dat', "rb") as f:
            word_overlap_matrix = pickle.load(f)
    else:
        word_inhib_matrix = build_word_inhibition_matrix(lexicon,lexicon_word_bigrams,pm,tokens_to_lexicon_indices)










