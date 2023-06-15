import pickle
import codecs
import pandas as pd
import os
import numpy as np
import chardet
import json
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer, set_seed
from reading_components import semantic_processing
from reading_helper_functions import build_word_inhibition_matrix
import logging
import spacy
import torch
from collections import defaultdict

logger = logging.getLogger(__name__)
# please make sure spacy model is downloaded using "python -m spacy download en_core_web_sm"
spacy_model = spacy.load("en_core_web_sm")

def get_stimulus_text_from_file(filepath, sep='\t'):

    encoding = chardet.detect(open(filepath, "rb").read())['encoding']
    stim_name = os.path.basename(filepath).replace('.txt', '').replace('.csv', '')

    if ".txt" in filepath:
        with codecs.open(filepath, encoding=encoding, errors='strict') as infile:
            text = infile.read()
            text = text.encode('UTF-8').decode('UTF-8')
            data = {'all': [text]}

    else:
        data = pd.read_csv(filepath, sep=sep, encoding=encoding)

    return data, stim_name

def pre_process_string(string, remove_punctuation=True, all_lowercase=True, strip_spaces=True, lemmatize=False):

    if remove_punctuation:
        string = re.sub(r'[^\w\s]', '', string)
    if all_lowercase:
        string = string.lower()
    if strip_spaces:
        string = string.strip()
    if lemmatize:
        if string:
            token = spacy_model(string)[0]
            if token:
                string = str(token.lemma_)
    return string

def create_freq_file(language, task_words, output_file_frequency_map, freq_threshold, n_high_freq_words, task, verbose):

    # TODO AL: this was needed to reproduce results on PSCall because the overlap between the words and SUBTLEX-DE was low (less than half). Need to fix this later
    if task == 'continuous reading' and language == 'german':
        filepath = "../data/frequency/PSCall_freq_pred.txt"
        my_data = pd.read_csv(filepath, delimiter="\t",
                              encoding=chardet.detect(open(filepath, "rb").read())['encoding'])
        my_data['word'] = my_data['word'].astype('unicode')
        file_freq_dict = dict()
        for word, freq in zip(my_data['word'].tolist(),my_data['f'].tolist()):
            word = pre_process_string(word)
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
        # preprocess words for correct matching with tokens in stimulus
        freq_words[word_col] = freq_words[word_col].apply(lambda x : pre_process_string(x))
        # only keep words whose frequencies are higher than threshold
        freq_words = freq_words[freq_words[freq_type] > freq_threshold]
        frequency_words_dict = dict(zip(freq_words[freq_words.columns[0]], freq_words[freq_words.columns[1]]))

        # create dict with frequencies from words in task stimuli which are also in the resource/corpus file
        file_freq_dict = {}
        overlapping_words = list(set(task_words) & set(freq_words[word_col].tolist()))
        for word in overlapping_words:
            file_freq_dict[word] = frequency_words_dict[word]

        # add top n words from frequency resource
        for line_number in range(n_high_freq_words):
            file_freq_dict[(freq_words.iloc[line_number][0])] = freq_words.iloc[line_number][1]

        if verbose:
            print("amount of words in task:", len(task_words))
            print("amount of overlapping words", len(overlapping_words))
            print('frequency file stored in ' + output_file_frequency_map)

        with open(output_file_frequency_map, "w") as f:
            json.dump(file_freq_dict, f, ensure_ascii=False)

def get_word_freq(pm, unique_words, n_high_freq_words = 500, freq_threshold = 0.15, verbose=True):

    output_word_frequency_map = f"../data/frequency/frequency_map_{pm.stim_name}_{pm.task_to_run}_{pm.language}.json"

    # AL: in case freq file needs to be created from original files
    if not os.path.exists(output_word_frequency_map):
        create_freq_file(pm.language, unique_words, output_word_frequency_map, freq_threshold, n_high_freq_words, pm.task_to_run, verbose)

    with open(output_word_frequency_map, "r") as f:
        word_freq_dict = json.load(f)

    return word_freq_dict


def create_pred_file(pm, output_file_pred_map, lexicon):

    word_pred_values_dict = dict()
    unknown_word_pred_values_dict = dict()

    if pm.prediction_flag == 'language model':

        # initialize language model and its tokenizer
        language_model = GPT2LMHeadModel.from_pretrained('gpt2')
        lm_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # in case you want to reproduce predictions
        set_seed(pm.seed)
        # list of words, set of words, sentences or passages. Each one is equivalent to one trial in an experiment
        for i, sequence in enumerate(pm.stim_all):
            sequence = [token for token in sequence.split(' ') if token != '']
            pred_dict = dict()
            unknown_tokens = dict()
            pred_info = semantic_processing(sequence, lm_tokenizer, language_model, pm.topk)
            for pos in range(1, len(sequence)):
                pred_dict[str(pos)] = dict()
                unknown_tokens[str(pos)] = dict()
                for token, pred in zip(pred_info[pos][0], pred_info[pos][1]):
                    token_processed = pre_process_string(token)
                    pred_dict[str(pos)][token_processed] = pred
                    if token_processed not in lexicon:
                        unknown_tokens[str(pos)][token] = pred
                    # else: # in case token is a sub-word, try to concatenate token with next predicted token
                    #     concat_string = ' '.join(sequence[:i]) + ' ' + token
                    #     input = lm_tokenizer(concat_string, return_tensors='pt')
                    #     output = language_model(**input)
                    #     logits = output.logits[:, -1, :]
                    #     pred_token = lm_tokenizer.decode([torch.argmax(logits).item()])
                    #     # pred_token = pre_process_string(pred_token)
                    #     merged_token = pre_process_string(token + pred_token, lemmatize = True)
                    #     if merged_token in lexicon:
                    #         pred_dict[str(pos)][merged_token] = round(pred, 3)
                    #         logger.info(f'{token} + {pred_token} = {merged_token}')
                    #     else:
                    #         unknown_tokens[str(pos)][token] = round(pred, 3)

            word_pred_values_dict[str(i)] = pred_dict
            unknown_word_pred_values_dict[str(i)] = unknown_tokens

        # logger.info('Predicting 1 subtoken')
        # logger.info('Unknown tokens predicted by gpt2: ' +
        #       str(sum([len(words.keys()) for text, info in unknown_word_pred_values_dict.items() if info for idx, words in info.items()])))
        # logger.info('Known tokens predicted by gpt2: ' +
        #       str(sum([len(words.keys()) for text, info in word_pred_values_dict.items() if info for idx, words in info.items()])))

    elif pm.prediction_flag == 'cloze':

        if 'psc' in pm.stim_name.lower():
            filepath = "../data/predictability/PSCall_freq_pred.txt"
            my_data = pd.read_csv(filepath, delimiter="\t",
                                  encoding=chardet.detect(open(filepath, "rb").read())['encoding'])
            word_pred_values_dict = np.array(my_data['pred'].tolist())

        elif 'provo' in pm.stim_name.lower():
            filepath = "../data/predictability/Provo_Corpus-Predictability_Norms.csv"
            my_data = pd.read_csv(filepath, delimiter=",",
                                  encoding=chardet.detect(open(filepath, "rb").read())['encoding'])

            for text_id, info in my_data.groupby(['Text_ID']):
                text_id = str(int(text_id) - 1)
                word_pred_values_dict[text_id] = dict()
                unknown_word_pred_values_dict[text_id] = dict()
                for text_position, responses in info.groupby(['Word_Number']):
                    responses = responses.to_dict('records')
                    word_pred_values_dict[text_id][str(int(text_position) - 1)] = dict()
                    unknown_word_pred_values_dict[text_id][str(int(text_position) - 1)] = dict()
                    for response in responses:
                        if response['Response'] and type(response['Response']) == str:
                            word = pre_process_string(response['Response'], lemmatize=pm.lemmatize)
                            word_pred_values_dict[text_id][str(int(text_position) - 1)][word] = float(response['Response_Proportion'])
                            if word not in lexicon:
                                unknown_word_pred_values_dict[text_id][str(int(text_position) - 1)][word] = float(response['Response_Proportion'])

            logger.info('Unknown tokens predicted in cloze task: ' +
                        str(sum([len(words.keys()) for text, info in unknown_word_pred_values_dict.items() if info for
                             idx, words in info.items()])))
            logger.info('Known tokens predicted in cloze task: ' +
                        str(sum([len(words.keys()) for text, info in word_pred_values_dict.items() if info for
                             idx, words in info.items()])))

    elif pm.prediction_flag == 'grammar':
        pass
        # if pm.task == 'continuous reading':
        #     with open("../data/PSCALLsyntax_probabilites.pkl", "r") as f:
        #         word_pred_values = np.array(pickle.load(f)["pred"].tolist())
        # else:
        #     grammar_prob_dt = pd.read_csv('../data/POSprob_' + pm.task + '.csv')
        #     grammar_prob = grammar_prob_dt.values.tolist()
        #     grammar_prob = np.array(grammar_prob)
        #     if pm.task_to_run == 'Sentence':
        #         word_pred_values = np.reshape(grammar_prob, (2, 400, 4))
        #     elif pm.task_to_run == 'Transposed':
        #         word_pred_values = np.reshape(grammar_prob, (2, 240, 5))
        #     elif pm.task_to_run == 'Classification':
        #         word_pred_values = np.reshape(grammar_prob, (2, 200, 3))
        #     else:
        #         raise NotImplementedError(f'Grammar probabilities not implemented for {pm.task} yet')

    else: # prediction_flag == uniform
        pass
        # word_pred_values = np.repeat(0.25, len(task_trials))

    with open(output_file_pred_map, "w") as f:
        json.dump(word_pred_values_dict, f, ensure_ascii=False)

    if unknown_word_pred_values_dict:
        output_file = output_file_pred_map.replace('.json', '') + '_unknown.json'
        with open(output_file, "w") as f:
            json.dump(unknown_word_pred_values_dict, f, ensure_ascii=False)

def get_pred_dict(pm, lexicon):

    output_word_pred_map = f"../data/predictability/prediction_map_{pm.stim_name}_{pm.prediction_flag}_{pm.task_to_run}_{pm.language}.json"
    if pm.prediction_flag == 'language model':
        output_word_pred_map = output_word_pred_map.replace('.json',f'_topk{pm.topk}.json')

    # AL: in case pred file needs to be created from original files
    if not os.path.exists(output_word_pred_map):
        create_pred_file(pm, output_word_pred_map, lexicon)

    with open(output_word_pred_map, "r") as f:
        word_pred_dict = json.load(f)

    return word_pred_dict

def check_previous_inhibition_matrix(pm,lexicon,lexicon_word_bigrams,verbose=False):

    inhib_matrix_parameters = '../data/inhib_matrix_params_latest_run.pkl'
    inhib_matrix_previous = '../data/inhibition_matrix_previous.pkl'

    if os.path.exists(inhib_matrix_parameters):

        with open(inhib_matrix_parameters, "rb") as f:
            parameters_previous = pickle.load(f)

        size_of_file = os.path.getsize(inhib_matrix_previous)

        # NV: compare the previous params with the actual ones.
        # the idea is that the matrix is fully dependent on these parameters alone.
        # So, if the parameters are the same, the matrix should be the same.
        # The file size is also added as a check
        if str(lexicon_word_bigrams)+str(len(lexicon)) +\
           str(pm.simil_algo)+str(pm.max_edit_dist) + str(pm.short_word_cutoff)+str(size_of_file) \
           == parameters_previous:
            previous_matrix_usable = True
        else:
            previous_matrix_usable = False
    else:
        previous_matrix_usable = False
        if verbose:
            print('no previous inhibition matrix')

    return previous_matrix_usable

def set_up_inhibition_matrix(pm, lexicon, lexicon_word_ngrams, tokens_to_lexicon_indices):

    previous_matrix_usable = check_previous_inhibition_matrix(pm,lexicon,lexicon_word_ngrams)

    if previous_matrix_usable:
        with open('../data/Inhibition_matrix_previous.pkl', "rb") as f:
            word_inhibition_matrix = pickle.load(f)
    else:
        word_inhibition_matrix = build_word_inhibition_matrix(lexicon,lexicon_word_ngrams,pm,tokens_to_lexicon_indices)

    return word_inhibition_matrix

def write_out_simulation_data(simulation_data,outfile_sim_data, type='fixated'):

    simulation_results = defaultdict(list)

    for sim_index, texts_simulations in simulation_data.items():
        for text_index, text in texts_simulations.items():
            if type == 'fixated':
                for fix_counter, fix_info in text.items():
                    simulation_results['fixation_counter'].append(fix_counter)
                    simulation_results['text_id'].append(text_index)
                    simulation_results['simulation_id'].append(sim_index)
                    for info_name, info_value in fix_info.items():
                        simulation_results[info_name].append(info_value)
            elif type == 'skipped':
                for skipped in text:
                    simulation_results['text_id'].append(text_index)
                    simulation_results['simulation_id'].append(sim_index)
                    for info_name, info_value in skipped.items():
                        simulation_results[info_name].append(info_value)

    simulation_results_df = pd.DataFrame.from_dict(simulation_results)
    simulation_results_df.to_csv(outfile_sim_data, sep='\t', index=False)

# def find_wordskips(all_data, tokens):
#
#     wordskip_indices = set()
#     counter = 0
#
#     word_indices = [fx['foveal word index'] for fx in all_data.values()]
#     saccades = [fx['saccade type'] for fx in all_data.values()]
#
#     for word_i, saccade in zip(word_indices, saccades):
#         if saccade == 'wordskip':
#             if counter - 1 > 0 and saccades[counter - 1] != 'regression':
#                 wordskip_indices.add(word_i - 1)
#         counter += 1
#
#     return wordskip_indices