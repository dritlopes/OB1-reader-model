import pickle
import codecs
import pandas as pd
import os
import numpy as np
import chardet
import json

def get_stimulus_text_from_file(filepath):

    encoding = chardet.detect(open(filepath, "rb").read())['encoding']
    if ".txt" in filepath:
        with codecs.open(filepath, encoding=encoding, errors='strict') as infile:
            text = infile.read()
            text = text.encode('UTF-8').decode('UTF-8')
            data = {'all': text}

    else:
        data = pd.read_csv(filepath,sep=',',encoding=encoding)

    return data

def create_freq_file(language, task_words, output_file_frequency_map, freq_threshold, n_high_freq_words, task, verbose):

    # this was needed to reproduce results on PSCall because the overlap between the words and SUBTLEX-DE was low (less than half). Need to fix this later
    if task == 'continuous reading' and language == 'german':
        filepath = "../data/PSCall_freq_pred.txt"
        my_data = pd.read_csv(filepath, delimiter="\t",
                              encoding=chardet.detect(open(filepath, "rb").read())['encoding'])
        my_data['word'] = my_data['word'].astype('unicode')
        file_freq_dict = dict()
        for word, freq in zip(my_data['word'].tolist(),my_data['f'].tolist()):
            word = word.strip().replace(".", "").replace(",","")
            file_freq_dict[word] = float(freq)
        with open(output_file_frequency_map, "w") as f:
            json.dump(file_freq_dict, f, ensure_ascii=False)

    else:

        if language == 'english':
            filepath = '../data/SUBTLEX_UK.txt'
            columns_to_use = [0, 1, 5]
            freq_type = 'LogFreq(Zipf)'
            word_col = 'Spelling'

        elif language == 'french':
            filepath = '../data/French_Lexicon_Project.txt'
            columns_to_use = [0, 7, 8, 9, 10]
            freq_type = 'cfreqmovies'
            word_col = 'Word'

        elif language == 'german':
            filepath = '../data/SUBTLEX_DE.txt'
            columns_to_use = [0, 1, 3, 4, 5, 9]
            freq_type = 'lgSUBTLEX'
            word_col = 'Word'

        elif language == 'dutch':
            filepath = '../data/SUBTLEX-NL.txt'
            columns_to_use = [0, 7]
            freq_type = 'Zipf'
            word_col = 'Word'

        else:
            raise NotImplementedError(language + " is not implemented yet!")

        # create dict of word frequencies from resource/corpus file
        freq_df = pd.read_csv(filepath, usecols=columns_to_use, dtype={word_col: np.dtype(str)},
                              encoding=chardet.detect(open(filepath, "rb").read())['encoding'], delimiter="\t")
        freq_df.sort_values(by=[freq_type], ascending=False, inplace=True, ignore_index=True)
        freq_df[word_col] = freq_df[word_col].astype('unicode')
        freq_words = freq_df[[word_col, freq_type]]
        # NV: convert to Zipf scale. # from frequency per million to zipf. Also, replace -inf with 1
        if freq_type == 'cfreqmovies':
            freq_words['cfreqmovies'] = freq_words['cfreqmovies'].apply(lambda x: np.log10(x * 1000) if x > 0 else 0)
        # convert strings to floats
        freq_words[freq_type] = freq_words[freq_type].replace(',', '.', regex=True).astype(float)
        # only keep words whose frequencies are higher than threshold
        freq_words = freq_words[freq_words[freq_type] > freq_threshold]
        frequency_words_dict = dict(zip(freq_words[freq_words.columns[0]], freq_words[freq_words.columns[1]]))

        # create dict with frequencies from words in task stimuli which are also in the resource/corpus file
        file_freq_dict = {}
        overlapping_words = list(set(task_words) & set(freq_words[word_col].tolist()))
        for word in overlapping_words:
            file_freq_dict[word] = frequency_words_dict[word]

        # only keep top n words
        for line_number in range(n_high_freq_words):
            file_freq_dict[((freq_words.iloc[line_number][0]).lower())] = freq_words.iloc[line_number][1]

        if verbose:
            print("amount of words in task:", len(task_words))
            print("amount of overlapping words", len(overlapping_words))
            print('frequency file stored in ' + output_file_frequency_map)

        with open(output_file_frequency_map, "w") as f:
            json.dump(file_freq_dict, f, ensure_ascii=False)

def get_word_freq(pm, unique_words, n_high_freq_words = 500, freq_threshold = 0.15, verbose=False):

    output_word_frequency_map = "../data/" + pm.task_to_run + "_frequency_map_" + pm.language + ".json"

    if not os.path.exists(output_word_frequency_map): #AL: in case freq file needs to be created from original files
        create_freq_file(pm.language, unique_words, output_word_frequency_map, freq_threshold, n_high_freq_words, pm.task_to_run, verbose)

    with open(output_word_frequency_map, "r") as f:
        word_freq_dict = json.load(f)

    return word_freq_dict


def create_pred_file(pm, task_words, output_file_pred_map):

    if pm.use_grammar_prob:
        if pm.task == 'continuous reading':
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
        if pm.task_to_run == 'continuous reading':
            filepath = "../data/PSCall_freq_pred.txt"
            my_data = pd.read_csv(filepath, delimiter="\t", encoding=chardet.detect(open(filepath, "rb").read())['encoding'])
            word_pred_values = np.array(my_data['pred'].tolist())
        else:
            word_pred_values = np.repeat(1, len(task_words))

    word_pred_values_dict = dict()
    for i, pred in enumerate(word_pred_values):
        word_pred_values_dict[i] = pred

    with open(output_file_pred_map, "w") as f:
        json.dump(word_pred_values_dict, f, ensure_ascii=False)

def get_pred_values(pm, task_words):

    output_word_pred_map = "../data/" + pm.task_to_run + "_predictions_map_" + pm.language + ".json"

    if not os.path.exists(output_word_pred_map):  # AL: in case pred file needs to be created from original files
        create_pred_file(pm,task_words,output_word_pred_map)

    with open(output_word_pred_map, "r") as f:
        word_pred_dict = json.load(f)

    return word_pred_dict

def check_previous_inhibition_matrix(pm,lexicon,lexicon_word_bigrams,verbose=False):

    if os.path.exists('../data/Inhib_matrix_params_latest_run.dat'):

        with open('../data/Inhib_matrix_params_latest_run.dat', "rb") as f:
            parameters_previous = pickle.load(f)

        size_of_file = os.path.getsize('../data/Inhibition_matrix_previous.dat')

        # NV: compare the previous params with the actual ones.
        # the idea is that the matrix is fully dependent on these parameters alone.
        # So, if the parameters are the same, the matrix should be the same.
        # The file size is also added as a check
        if str(lexicon_word_bigrams)+str(len(lexicon))+str(pm.min_overlap) +\
           str(pm.simil_algo)+str(pm.max_edit_dist) + str(pm.short_word_cutoff)+str(size_of_file) \
           == parameters_previous:
            previous_matrix_usable = True
        else:
            previous_matrix_usable = False
    else:
        previous_matrix_usable = False
        if verbose: print('no previous inhibition matrix')

    return previous_matrix_usable

