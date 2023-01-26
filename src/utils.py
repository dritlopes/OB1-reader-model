import pickle
import codecs
import pandas as pd
import os
import numpy as np
import chardet


def get_stimulus_text_from_file(filepath):

    if ".pkl" in filepath:
        with open(filepath) as infile:
            data = pickle.load(infile)

    elif ".txt" in filepath:
        with codecs.open(filepath, encoding='ISO-8859-1', errors='strict') as infile:
            text = infile.read().lower()
            data = {'all': text}
    else:
        data = pd.read_csv(filepath,sep=';')

    return data


def create_freq_file(language, task_words, output_file_frequency_map, freq_threshold, n_high_freq_words, verbose=False):

    if language == 'english':
        filepath = '../data/SUBTLEX_UK.txt'
        columns_to_use = [0, 1, 5]
        freq_type = 'LogFreq(Zipf)'

    elif language == 'french':
        filepath = '../data/French_Lexicon_Project.txt'
        columns_to_use = [0, 7, 8, 9, 10]
        freq_type = 'cfreqmovies'

    elif language == 'german':
        filepath = '../data/SUBTLEX_DE.txt'
        columns_to_use = [0, 1, 3, 4, 5, 9]
        freq_type = 'lgSUBTLEX'

    elif language == 'dutch':
        filepath = '../data/SUBTLEX-NL.txt'
        columns_to_use = [0, 7]
        freq_type = 'Zipf'

    else:
        raise NotImplementedError(language + " is not implemented yet!")

    #AL: create dict of word frequencies from resource/corpus file
    freqlist_arrays = pd.read_csv(filepath,
                                  usecols=columns_to_use, dtype={'Word': np.dtype(str)},
                                  encoding=chardet.detect(open(filepath, "rb").read())['encoding'], delimiter="\t")
    freqlist_arrays.sort_values(by=[freq_type], ascending=False, inplace=True, ignore_index=True)
    #AL: only keep words whose frequencies are higher than threshold
    freqlist_arrays = freqlist_arrays[freqlist_arrays[freq_type] > freq_threshold]
    freq_words = freqlist_arrays[['Word', freq_type]]
    frequency_words_dict = dict(zip(freq_words[freq_words.columns[0]], freq_words[freq_words.columns[1]]))
    frequency_words_np = np.array(freq_words)

    #AL: create dict with frequencies from words in task stimuli which are also in the resource/corpus file
    file_freq_dict = {}
    overlapping_words = list(set(task_words) & set(frequency_words_np))
    for word in overlapping_words:
        file_freq_dict[word] = frequency_words_dict[word]

    #AL: only keep top n words
    for line_number in range(n_high_freq_words):
        file_freq_dict[((freq_words.iloc[line_number][0]).lower())] = freq_words.iloc[line_number][1]

    with open(output_file_frequency_map, "wb") as f:
        pickle.dump(file_freq_dict, f)

    if verbose:
        print("words in task:\n", task_words)
        print("amount of words in task:", len(task_words))
        print("words in task AND in dictionary:\n", overlapping_words)
        print("amount of overlapping words", len(overlapping_words))
        print('frequency file stored in ' + output_file_frequency_map)


def get_word_freq(pm, unique_words, n_high_freq_words = 500, freq_threshold = 0.15, verbose=False):

    output_word_frequency_map = "../data/" + pm.task_to_run + "_frequency_map_" + pm.short[pm.language] + ".dat"

    if not os.path.exists(output_word_frequency_map): #AL: in case freq file needs to be created from original files
        create_freq_file(pm.language, unique_words, output_word_frequency_map, freq_threshold, n_high_freq_words, verbose)

    with open(output_word_frequency_map, "rb") as f:
        word_freq_dict = pickle.load(f, encoding="latin1")  # For Python3

    return word_freq_dict


def create_pred_file(pm, task_words, output_file_pred_map):

    if pm.use_grammar_prob:
        if pm.task == 'PSCall':
            with open("../data/PSCALLsyntax_probabilites.pkl", "r") as f:
                word_pred_values = np.array(pickle.load(f)["pred"].tolist())
        else:
            grammar_prob_dt = pd.read_csv('../data/POSprob_' + pm.task + '.csv')
            grammar_prob = grammar_prob_dt.values.tolist()
            grammar_prob = np.array(grammar_prob)
            if pm.task_to_run == 'Sentence':
                word_pred_values = np.reshape(grammar_prob, (2, 400, 4))
            elif pm.task_to_run == 'Transposed':
                word_pred_values = np.reshape(grammar_prob, (2, 240, 5))
            elif pm.task_to_run == 'Classification':
                word_pred_values = np.reshape(grammar_prob, (2, 200, 3))
            else:
                raise NotImplementedError(f'Grammar probabilities not implemented for {pm.task} yet')
    elif pm.uniform_prob:
        word_pred_values = np.repeat(0.25, len(task_words))
    else:
        if pm.task == 'PSCall':
            my_data = pd.read_csv("../data/PSCall_freq_pred.txt", delimiter="\t", encoding="ISO-8859-1")
            word_pred_values = np.array(my_data['pred'].tolist())
        else:
            word_pred_values = np.repeat(1, len(task_words))

    with open(output_file_pred_map, "wb") as f:
        pickle.dump(word_pred_values, f)


def get_pred_values(pm, task_words):

    output_word_pred_map = "../data/" + pm.task_to_run + "_predictions_map_" + pm.short[pm.language] + ".dat"

    if not os.path.exists(output_word_pred_map):  # AL: in case pred file needs to be created from original files
        create_pred_file(pm,task_words,output_word_pred_map)

    with open(output_word_pred_map, "rb") as f:
        word_pred_dict = pickle.load(f, encoding="latin1")  # For Python3

    return word_pred_dict


def check_previous_inhibition_matrix(pm,lexicon,lexicon_word_bigrams,verbose=False):

    if os.path.exists('../data/Inhib_matrix_params_latest_run.dat'):

        with open('../data/Inhib_matrix_params_latest_run.dat', "rb") as f:
            parameters_previous = pickle.load(f)

        size_of_file = os.path.getsize('..data/Inhibition_matrix_previous.dat')
        # NV: compare the previous params with the actual ones.
        # he idea is that the matrix is fully dependent on these parameters alone.
        # So, if the parameters are the same, the matrix should be the same.
        # The file size is also added as a check . Note: Could possibly be more elegant
        if str(lexicon_word_bigrams)+str(len(lexicon))+str(pm.min_overlap) +\
           str(len(lexicon))+str(pm.affix_system) +\
           str(pm.simil_algo)+str(pm.max_edit_dist) + str(pm.short_word_cutoff)+str(size_of_file) \
           == parameters_previous:
            previous_matrix_usable = True
        else:
            previous_matrix_usable = False
    else:
        previous_matrix_usable = False
        if verbose: print('no previous inhibition matrix')

    return previous_matrix_usable



