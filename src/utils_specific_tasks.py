import pickle
import codecs
import pandas as pd
import os
import numpy as np
import chardet
import json
import re
import seaborn as sns
import matplotlib as plt
from reading_components import match_active_words_to_input_slots
from collections import defaultdict
from reading_helper_functions import build_word_inhibition_matrix, cal_ngram_exc_input, get_threshold
from reading_components import semantic_processing
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer

def get_stimulus_text_from_file(filepath, sep='\t'):

    #encoding = chardet.detect(open(filepath, "rb").read())['encoding']
    stim_name = os.path.basename(filepath).replace('.txt', '').replace('.csv', '')

    # if ".txt" in filepath:
    #     with codecs.open(filepath, encoding=encoding, errors='strict') as infile:
    #         text = infile.read()
    #         text = text.encode('UTF-8').decode('UTF-8')
    #         data = {'all': text}
    if ".txt" in filepath:
        encoding = chardet.detect(open(filepath, "rb").read())['encoding']
        with codecs.open(filepath, encoding=encoding, errors='strict') as infile:
            text = infile.read()
            text = text.encode('UTF-8').decode('UTF-8')
            data = {'all': [text]}

    else:
        # data = pd.read_csv(filepath,sep=';',encoding=encoding)
        data = pd.read_csv(filepath, sep=sep, encoding="ISO-8859-1")

        if stim_name == 'Provo_Corpus-Predictability_Norms':
            ids, texts, words, word_ids = [], [], [], []
            for i, text_info in data.groupby('Text_ID'):
                ids.append(int(i)-1)
                text = text_info['Text'].tolist()[0]
                # fix error in text 36 in raw data
                if int(i) == 36:
                    text = text.replace(' Ñ', '')
                texts.append(text)
                text_words = [pre_process_string(token) for token in text.split()]
                text_word_ids = [i for i in range(0,len(text_words))]
                words.append(text_words)
                word_ids.append(text_word_ids)
            data = pd.DataFrame(data={'id': ids,
                                      'all': texts,
                                      'words': words,
                                      'word_ids': word_ids})

    return data, stim_name

def pre_process_string(string, remove_punctuation=True, all_lowercase=True, strip_spaces=True):

    if remove_punctuation:
        string = re.sub(r'[^\w\s]', '', string)
    if all_lowercase:
        string = string.lower()
    if strip_spaces:
        string = string.strip()
    return string

def create_freq_file(language, task_words, output_file_frequency_map, freq_threshold, n_high_freq_words, task, verbose):

    # TODO AL: this was needed to reproduce results on PSCall because the overlap between the words and SUBTLEX-DE was low (less than half). Need to fix this later
    if task == 'continuous_reading' and language == 'german':
        filepath = "data\PSCall_freq_pred.txt"
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
            filepath = 'data\SUBTLEX_UK.txt'
            columns_to_use = [0, 1, 5]
            freq_type = 'LogFreq(Zipf)'
            word_col = 'Spelling'

        elif language == 'french':
            filepath = 'data\French_Lexicon_Project.txt'
            columns_to_use = [0, 7, 8, 9, 10]
            freq_type = 'cfreqmovies'
            word_col = 'Word'

        elif language == 'german':
            filepath = 'data\SUBTLEX_DE.txt'
            columns_to_use = [0, 1, 3, 4, 5, 9]
            freq_type = 'lgSUBTLEX'
            word_col = 'Word'

        elif language == 'dutch':
            filepath = 'data\SUBTLEX-NL.txt'
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

def get_word_freq(pm, unique_words, n_high_freq_words = 500, freq_threshold = 0.15, verbose=False):

    output_word_frequency_map = f"data\{pm.task_to_run}_{pm.stim_name}_frequency_map_{pm.language}.json"

    # AL: in case freq file needs to be created from original files
    if not os.path.exists(output_word_frequency_map):
        create_freq_file(pm.language, unique_words, output_word_frequency_map, freq_threshold, n_high_freq_words, pm.task_to_run, verbose)

    with open(output_word_frequency_map, "r") as f:
        word_freq_dict = json.load(f)
    
    #print(f"Loaded word frequency dictionary: {word_freq_dict}")

    return word_freq_dict


def create_pred_file(pm, output_file_pred_map, lexicon):

    word_pred_values_dict = dict()
    unknown_word_pred_values_dict = dict()

    if pm.prediction_flag in ['gpt2', 'llama']:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device ', device)

        # load language model and its tokenizer
        if pm.prediction_flag == 'gpt2':
            language_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
            lm_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        elif pm.prediction_flag == 'llama':
            # torch_dtype=torch.float16 is important to load model directly on gpu and to decrease the needed ram
            # https://huggingface.co/decapoda-research/llama-7b-hf/discussions/2
            # https://discuss.huggingface.co/t/llama-7b-gpu-memory-requirement/34323
            language_model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf", torch_dtype=torch.float16).to(device)
            # !!! after downloading the tokenizer, please go to where downloaded model is located
            # ($HOME/<username>/.cache/huggingface/<...>) and change tokenizer config file as done here:
            # https://huggingface.co/decapoda-research/llama-7b-hf/discussions/103/files
            # legacy=false makes sure fix in handling of tokens read after a special token is used,
            # see https://github.com/huggingface/transformers/pull/24565 for more details
            lm_tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf", legacy=False)
            # Additional info when using cuda
            if device.type == 'cuda':
                print(torch.cuda.get_device_name(0))
                print('Memory Usage:')
                print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
                print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

        # list of words, set of words, sentences or passages. Each one is equivalent to one trial in an experiment
        for i, sequence in enumerate(pm.stim_all):
            sequence = [token for token in sequence.split(' ') if token != '']
            pred_dict = dict()
            unknown_tokens = dict()
            pred_info = semantic_processing(sequence, lm_tokenizer, language_model, pm.prediction_flag, pm.topk, pm.pred_threshold, device)
            for pos in range(1, len(sequence)):
                target = pre_process_string(sequence[pos])
                pred_dict[str(pos)] = {'target': target,
                                        'predictions': dict()}
                unknown_tokens[str(pos)] = {'target': target,
                                            'predictions': dict()}
                for token, pred in zip(pred_info[pos][0], pred_info[pos][1]):
                    token_processed = pre_process_string(token)
                    # language models may use uppercase, while our lexicon only has lowercase, so take the higher pred
                    if token_processed not in pred_dict[str(pos)]['predictions'].keys():
                        pred_dict[str(pos)]['predictions'][token_processed] = pred
                    if token_processed not in lexicon:
                        unknown_tokens[str(pos)]['predictions'][token] = {'token_processed': token_processed,
                                                                          'pred': pred}
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
            filepath = "../data/raw/PSCall_freq_pred.txt"
            my_data = pd.read_csv(filepath, delimiter="\t",
                                  encoding=chardet.detect(open(filepath, "rb").read())['encoding'])
            word_pred_values_dict = np.array(my_data['pred'].tolist())

        elif 'provo' in pm.stim_name.lower():

            # encoding = chardet.detect(open(filepath, "rb").read())['encoding']
            filepath = "data\raw\Provo_Corpus-Predictability_Norms.csv"
            my_data = pd.read_csv(filepath, encoding="ISO-8859-1")
            # align indexing with ob1 stimuli (which starts at 0, not at 1)
            my_data['Text_ID'] = my_data['Text_ID'].apply(lambda x : str(int(x)-1))
            my_data['Word_Number'] = my_data['Word_Number'].apply(lambda x : str(int(x)-1))
            # fix misplaced row in raw data
            my_data.loc[(my_data['Word_Number'] == '2') & (my_data['Word'] == 'probably') & (my_data['Text_ID'] == '17'), 'Text_ID'] = '54'

            for text_id, info in my_data.groupby('Text_ID'):

                word_pred_values_dict[text_id] = dict()
                unknown_word_pred_values_dict[text_id] = dict()

                for text_position, responses in info.groupby('Word_Number'):

                    # fix error in provo cloze data indexing
                    for row in [(2,44),(12,18)]:
                        if int(text_id) == row[0] \
                                and int(text_position) in range(row[1]+1, len(info['Word_Number'].unique())+2):
                            text_position = str(int(text_position) - 1)

                    target = pre_process_string(responses['Word'].tolist()[0])
                    word_pred_values_dict[text_id][text_position] = {'target': target,
                                                                     'predictions': dict()}
                    unknown_word_pred_values_dict[text_id][text_position] = {'target': target,
                                                                             'predictions': dict()}
                    responses = responses.to_dict('records')
                    for response in responses:
                        if response['Response'] and type(response['Response']) == str:
                            word = pre_process_string(response['Response'])
                            word_pred_values_dict[text_id][text_position]['predictions'][word] = float(response['Response_Proportion'])
                            if word not in lexicon:
                                unknown_word_pred_values_dict[text_id][text_position]['predictions'][response['Response']] = {'token_processed': word,
                                                                                                                            'pred': float(response['Response_Proportion'])}

            # check alignment between cloze target words and model stimuli words
            for i, text in pm.stim.iterrows():
                cloze_text = word_pred_values_dict[str(i)]
                text_words = text['words'].replace('[','').replace(']','').replace(',','').replace("'", "").split()
                text_word_ids = text['word_ids'].replace('[','').replace(']','').replace(',','').replace("'", "").split()
                for word, word_id in zip(text_words[1:], text_word_ids[1:]):
                    if (i, word_id) != (17, '50'): # word 50 in text 17 (last word of text 17) is missing from cloze data
                        try:
                            cloze_word = cloze_text[word_id]['target']
                            if word != cloze_word:
                                warnings.warn(f'Target word in cloze task "{cloze_word}" not the same as target word in model stimuli "{word}" in text {i}, position {word_id}')
                        except KeyError:
                            print(f'Position {word_id}, "{word}", in text {i} not found in cloze task')

            # logger.info('Unknown tokens predicted in cloze task: ' +
            #             str(sum([len(words.keys()) for text, info in unknown_word_pred_values_dict.items() if info for
            #                  idx, words in info.items()])))
            # logger.info('Known tokens predicted in cloze task: ' +
            #             str(sum([len(words.keys()) for text, info in word_pred_values_dict.items() if info for
            #                  idx, words in info.items()])))

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

# def get_pred_dict(pm, lexicon):

#     output_word_pred_map = f"../data/processed/prediction_map_{pm.stim_name}_{pm.prediction_flag}_{pm.task_to_run}_{pm.language}.json"
#     if pm.prediction_flag in ['gpt2', 'llama']:
#         output_word_pred_map = output_word_pred_map.replace('.json', f'_topk{pm.topk}.json')

#     # AL: in case pred file needs to be created from original files
#     if not os.path.exists(output_word_pred_map):
#         create_pred_file(pm, output_word_pred_map, lexicon)

#     with open(output_word_pred_map, "r") as f:
#         word_pred_dict = json.load(f)

#     return word_pred_dict

def get_pred_values(pm, task_words):

    output_word_pred_map = f"data\{pm.task_to_run}_{pm.stim_name}_prediction_map_{pm.language}.json"

    # AL: in case pred file needs to be created from original files
    if not os.path.exists(output_word_pred_map):
        create_pred_file(pm,task_words,output_word_pred_map)

    with open(output_word_pred_map, "r") as f:
        word_pred_dict = json.load(f)

    return word_pred_dict

def check_previous_inhibition_matrix(pm,lexicon,lexicon_word_bigrams,inhib_matrix_previous,inhib_matrix_parameters):

    previous_matrix_usable = False

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

    if not previous_matrix_usable:
        print('no previous inhibition matrix')

    return previous_matrix_usable

# def is_similar_word_length(len_word1, len_word2):
#     return abs(len_word1 - len_word2) < (0.25 * max(len_word1, len_word2))

def combine_affix_frequency_files(language):
    """
    Takes pm.language as input and combines the prefix and suffix frequency dictionaries into a single affix frequency dictionary.
    Combined dictionary is saved to a pickled file with a name based on the language code.
    If the combined affix frequency file already exists, the function does nothing.
    """
    # language codes for the files
    language_codes = {
        'english': 'en',
        'french': 'fr',
        'german': 'de'
    }
    language_code = language_codes.get(language.lower())
    combined_file_name = f'data/affix_frequency_{language_code}.dat'
    if os.path.exists(combined_file_name):
        return
    with open(f'data/suffix_frequency_{language_code}.dat', 'rb') as f:
        suffix_freq = pickle.load(f)
    with open(f'data/prefix_frequency_{language_code}.dat', 'rb') as f:
        prefix_freq = pickle.load(f)
    # Combine the suffix and prefix frequency dictionaries
    affix_freq = {**suffix_freq, **prefix_freq}
    with open(combined_file_name, 'wb') as f:
        pickle.dump(affix_freq, f)

def get_inhibition_value(stimulus, target, lexicon, word_inhibition_matrix, complex_stem_pairs):
    # inhibition_value = None
    # for word1, word2 in complex_stem_pairs:
    #     if word1 in stimulus.split() and word2 == target:
    #         i = lexicon.index(word1)
    #         j = lexicon.index(word2)
    #         inhibition_value = word_inhibition_matrix[i, j]
    #         #print(f'Complex pair: {word1}, {word2}')
    #         #print(f"Complex stem pair in trial: {word1}, {word2}; Inhibition value: {inhibition_value}")
    #         break
    # return inhibition_value, word1, word2

    # inhibition_value = 0  # Initialize inhibition_value to 0
    # word1, word2 = None, None
    # is_complex_pair = False

    # for pair in complex_stem_pairs:
    #     if pair[0] in stimulus.split() and pair[1] == target:
    #         i = lexicon.index(pair[0])
    #         j = lexicon.index(pair[1])
    #         inhibition_value = word_inhibition_matrix[i, j]
    #         word1, word2 = pair
    #         is_complex_pair = True
    #         #print(f'Complex pair: {word1}, {word2}')
    #         #print(f"Complex stem pair in trial: {word1}, {word2}; Inhibition value: {inhibition_value}")
    #         break
    # return inhibition_value, word1, word2, is_complex_pair


    
    i = lexicon.index(stimulus)
    j = lexicon.index(target)
    inhibition_value = word_inhibition_matrix[i, j]
    
    word1, word2 = None, None
    is_complex_pair = False

    for pair in complex_stem_pairs:
        if pair[0] in stimulus.split() and pair[1] == target:
            word1, word2 = pair
            is_complex_pair = True
            break
            
    return inhibition_value, word1, word2, is_complex_pair


def get_suffix_file(language): 
    # language codes for the files
    language_codes = {
        'english': 'en',
        'french': 'fr',
        'german': 'de'
    }
    language_code = language_codes.get(language.lower())
    file=f'data/suffix_frequency_{language_code}.dat'
    if os.path.exists(file):
        with open (file,"rb") as f:
            suffix_freq_dict = pickle.load(f)
        return suffix_freq_dict
    else:
        raise FileNotFoundError(f"The file {file} does not exist.")
    
def process_words_underscores(word_list):
    new_words = []
    lengths = []

    for word in word_list:
        new_word = f"_{word.strip().lower()}_"
        new_words.append(new_word)
        lengths.append(len(new_word))

    return new_words, lengths

def save_lexicon(lexicon, task):
    lexicon_file_name = 'Data/Lexicon_'+task+'.dat'
    with open(lexicon_file_name, "wb") as f:
        pickle.dump(lexicon, f)

def map_stimuli_words_to_lexicon_indices(individual_words, lexicon):
    # Initialize numpy array of zeros
    individual_to_lexicon_indices = np.zeros((len(individual_words)), dtype=int)
    
    # Populate the numpy array with indices
    for i, word in enumerate(individual_words):
        individual_to_lexicon_indices[i] = lexicon.index(word)
    
    # Return the resulting numpy array
    return individual_to_lexicon_indices

def lexicon_thresholds_and_indeces(lexicon, word_freq_dict, max_frequency, wordfreq_p, max_threshold):
    # Initialize numpy array, dict for indices and dict for word activity
    lexicon_thresholds_np = np.zeros(len(lexicon))
    lexicon_index_dict = {}
    lexicon_word_activity = {}
    
    for i, word in enumerate(lexicon):
        # Get the threshold for the word
        lexicon_thresholds_np[i] = get_threshold(word, word_freq_dict, max_frequency, wordfreq_p, max_threshold)
        
        # Assign the index to the word
        lexicon_index_dict[word] = i
        
        # Initialize word activity for the word
        lexicon_word_activity[word] = 0.0

        # individual_to_lexicon_indices = np.zeros((len(individual_words)), dtype=int)
        # for i, word in enumerate(individual_words):
        #     individual_to_lexicon_indices[i] = lexicon.index(word)

    # Return the numpy array and dictionaries
    return lexicon_thresholds_np, lexicon_index_dict, lexicon_word_activity

def stimuli_word_threshold_dict(words, word_frequencies, max_frequency, wordfreq_p, max_threshold, value_to_insert):
    word_thresh_dict = {}
    # print(f"INDIVIDUAL WORDS: {words}")
    # print(f"INDIVIDUAL WORDS LEN: {len(words)}\n")
    # print(f"LEN: {len(word_frequencies)} WORD FREQ: {word_frequencies}\n")
    # print(f"VALUE TO INSERT OUTSIDE OF LOOP CODE 1: {value_to_insert}\n")
    
    for word in words:
        # print(f"WORD: {word}\n")
        word_thresh_dict[word] = get_threshold(word, word_frequencies, max_frequency, wordfreq_p, max_threshold)
        # print(f"WORD: {word} THRESHOLD: {word_thresh_dict[word]}\n")
        try:
            word_frequencies[word]
        except KeyError:
            word_frequencies[word] = value_to_insert
            # print(f"VALUE TO INSERT INSIDE THE LOOP CODE 1: {value_to_insert}\n")
    return word_thresh_dict


def handle_task_specifics(task, stimulus, stim, trial, all_data, pm, prime=None):
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


def calculate_unit_activations(all_ngrams, bigrams_to_locations, eye_position, AttentionPosition, pm, attend_width):
    """
    Calculates the activations for each n-gram based on their locations, the current eye position, and attention parameters.

    Parameters:
    - all_ngrams (list of str): A list containing n-grams (combinations of characters).
    - bigrams_to_locations (dict): A dictionary mapping n-grams to their respective locations and weights.
    - eye_position (float): The current position of the eye.
    - AttentionPosition (float): The center of the attentional distribution.
    - pm (object): An object containing model parameters, including 'letPerDeg' and 'attention_skew'.
    - attend_width (float): The width of the attentional distribution.

    Returns:
    - unit_activations (dict): A dictionary mapping each n-gram to its calculated activation value.

    Description:
    The function computes the activation of each n-gram based on its locations, weights, and attentional parameters. 
    The activation of an n-gram is influenced by its proximity to the eye position and the center of the attentional distribution. 
    The 'cal_ngram_exc_input' function is used to compute the excitatory input for each n-gram based on these parameters.
    """
    unit_activations = {}
    for ngram in all_ngrams:
        unit_activations[ngram] = 0
        locations = bigrams_to_locations[ngram]
        for ngram_location_and_weight in locations:
            ngram_location = ngram_location_and_weight[:len(ngram)]
            ngram_weight = ngram_location_and_weight[-1]
            unit_activations[ngram] += cal_ngram_exc_input(ngram_location,
                                                          ngram_weight,
                                                          eye_position,
                                                          AttentionPosition,
                                                          attend_width,
                                                          pm.letPerDeg,
                                                          pm.attention_skew)
    return unit_activations

    
    # This is where input is computed (excit is specific to word, inhib same for all)
    for lexicon_ix, lexicon_word in enumerate(lexicon):  # NS: why is this?
        wordExcitationInput = 0

        # (Fast) Bigram & Monogram activations
        bigram_intersect_list = allBigrams_set.intersection(
            lexicon_word_bigrams[lexicon_word])
        for bigram in bigram_intersect_list:
            wordExcitationInput += pm.bigram_to_word_excitation * \
                unitActivations[bigram]
        for monogram in allMonograms:
            if monogram in lexicon_word:
                wordExcitationInput += pm.bigram_to_word_excitation * \
                    unitActivations[monogram]

        word_input_np[lexicon_ix] = wordExcitationInput + wordBigramsInhibitionInput
    
    return word_input_np


def inter_word_inhibition(lexicon_word_activity_np, lexicon_total_input_np, lexicon_activewords_np, word_input_np, pm, word_overlap_matrix, lexicon_normalized_word_inhibition, N_ngrams_lexicon, lexicon, target):
    
    lexicon_word_activity_np_local = lexicon_word_activity_np.copy()
    lexicon_total_input_np_local = lexicon_total_input_np.copy()
    lexicon_activewords_np_local = lexicon_activewords_np.copy()

    # Active words selection vector (makes computations efficient)
    lexicon_activewords_np_local[(lexicon_word_activity_np_local > 0.0) | (word_input_np > 0.0)] = True

    # Calculate total inhibition for each word
    # Matrix * Vector (4x faster than vector)
    overlap_select = word_overlap_matrix[:, (lexicon_activewords_np_local == True)]

    indices_where_true = np.where(overlap_select)
    # print(f"WHERE TRUE: {indices_where_true}")
    # print(f"LEN OF TRUE: {len(indices_where_true)}")

    lexicon_select = (lexicon_word_activity_np_local + word_input_np)[(lexicon_activewords_np_local == True)] * lexicon_normalized_word_inhibition
    # print(f"lexicon word activity np: {lexicon_word_activity_np}")
    # print(f"word input np: {word_input_np}")
    # print(f"lexicon_activewords_np {lexicon_activewords_np}")
    # print(f"lexicon normalized word inhibition: {lexicon_normalized_word_inhibition}")

    lexicon_word_inhibition_np = np.dot((overlap_select**2), -(lexicon_select**2)) / np.array(N_ngrams_lexicon)
    

    lexicon_total_input_np_local = np.add(lexicon_word_inhibition_np, word_input_np)

    lexicon_word_activity_change = ((pm.max_activity - lexicon_word_activity_np_local) * lexicon_total_input_np_local) + \
                                    ((lexicon_word_activity_np_local - pm.min_activity) * pm.decay)
    # print(f"lexicon word activity change: {lexicon_word_activity_change}")
    lexicon_word_activity_np_local = np.add(lexicon_word_activity_np_local, lexicon_word_activity_change)

    # Correct activity beyond minimum and maximum activity to min and max
    lexicon_word_activity_np_local[lexicon_word_activity_np_local < pm.min_activity] = pm.min_activity
    lexicon_word_activity_np_local[lexicon_word_activity_np_local > pm.max_activity] = pm.max_activity
    
    squared_activity = np.square(lexicon_word_activity_np_local)

    # target_lexicon_index = [idx for idx, element in enumerate(lexicon) if element == '_'+target+'_']

    return lexicon_word_activity_np_local, lexicon_total_input_np_local, squared_activity, lexicon_word_inhibition_np, overlap_select, lexicon_select#, target_lexicon_index


# def match_words_in_slots(trial,recognized, n_words_in_stim, order_match_check, stim_matched_slots, stimulus, above_thresh_lexicon_np, lexicon, 
#                         affixes, lexicon_word_activity_np, pm, target, task, lexicon_index_dict):
#     new_recognized_words = np.zeros(len(lexicon), dtype=bool)
#     falseguess = False
#     noun_count = 0
#     ver_count = 0
#     POSrecognition = ['' for _ in range(n_words_in_stim)]

#     for slot_to_check in range(0, n_words_in_stim):
#         slot_num = order_match_check[slot_to_check]
#         # print(f"Order match check: {order_match_check}")
#         if len(stim_matched_slots[slot_num]) == 0:
#             word_searched = stimulus.split()[slot_num]
#             recognWrdsFittingLen_np = above_thresh_lexicon_np * \
#                         np.array([0 if x in affixes else int(is_similar_word_length(pm, len(x.replace('_', '')),
#                               len(word_searched))) for x in lexicon])
#             if sum(recognWrdsFittingLen_np):
#                 highest = np.argmax(recognWrdsFittingLen_np * lexicon_word_activity_np)
#                 highest_word = lexicon[highest]
#                 stim_matched_slots[slot_num] = highest_word
#                 new_recognized_words[highest] = 1
#                 above_thresh_lexicon_np[highest] = 0
#                 lexicon_word_activity_np[highest] = pm.min_activity
#                 if target in stimulus.split():
#                     if stimulus.split().index(target) == slot_num:
#                         if target == highest_word.replace('_', ''):
#                             recognized = True
#                         else:
#                             falseguess = True
#                 # if pm.use_grammar_prob:
#                 #     POSrecognition[slot_num] = POSdict[highest_word.replace('_', '')]
#                 #     if task == 'Classification':
#                 #         if POSrecognition[0] == 'NOU' or POSrecognition[2] == 'NOU':
#                 #             noun_count += lexicon_word_activity_np[lexicon_index_dict[
#                 #                 f'_{stimulus.split()[ slot_num]}_']]
#                 #         elif POSrecognition[0] == 'VER' or POSrecognition[2] == 'VER':
#                 #             ver_count += lexicon_word_activity_np[lexicon_index_dict[
#                 #                 f'_{stimulus.split()[ slot_num]}_']]
#                 #     if POSrecognition[slot_num] == POSdict[stimulus.split()[slot_num]]:
#                 #         if slot_num > 0:
#                 #             lexicon_word_activity_np[lexicon_index_dict[f'_{stimulus.split()[slot_num - 1]}_']]\
#                 #                 += word_pred_values[0][trial][slot_num - 1] * grammar_weight
#                 #         if slot_num < len(stimulus.split())-1:
#                 #             lexicon_word_activity_np[lexicon_index_dict[f'_{stimulus.split()[slot_num + 1]}_']] \
#                 #                 += word_pred_values[1][trial][slot_num + 1] * grammar_weight
#                 ### ADD OTHER TASKS HERE

    return stim_matched_slots, new_recognized_words, above_thresh_lexicon_np, lexicon_word_activity_np, recognized, falseguess, POSrecognition, noun_count, ver_count

def plot_inhib_spectrum(lexicon, lexicon_activewords_np, inhib_spectrum1, inhib_spectrum2,
                        index_num1, index_num2, inhib_spectrum1_indices, inhib_spectrum2_indices, 
                        cur_cycle):

    if any(lexicon_activewords_np):
        fig, axes = plt.subplots(1, 2)
        fig.set_size_inches(15, 7)
        
        x_coord=0
        
        for (IS, index_num, indices) in [(inhib_spectrum1,index_num1, inhib_spectrum1_indices) , (inhib_spectrum2, index_num2, inhib_spectrum2_indices)]:
        
            sns.stripplot(ax=axes[x_coord],
                          x=np.array(lexicon)[lexicon_activewords_np == True][indices][:10], 
                          y = IS[:10])
            axes[x_coord].set_title(lexicon[index_num])
            x_coord+=1
            
        plt.suptitle(f'{cur_cycle = }')
        plt.show()
    else:
        pass

def set_up_inhibition_matrix(pm, lexicon, lexicon_word_ngrams):

    matrix_filepath = 'data/processed/inhibition_matrix_previous.pkl'
    matrix_parameters_filepath = 'data/processed/inhibition_matrix_parameters_previous.pkl'

    previous_matrix_usable = check_previous_inhibition_matrix(pm, lexicon,
                                                              lexicon_word_ngrams,
                                                              matrix_filepath,
                                                              matrix_parameters_filepath)

    if previous_matrix_usable:
        with open(matrix_filepath, "rb") as f:
            word_inhibition_matrix = pickle.load(f)
    else:
        word_inhibition_matrix = build_word_inhibition_matrix(lexicon,
                                                              lexicon_word_ngrams,
                                                              pm,
                                                              matrix_filepath,
                                                              matrix_parameters_filepath)

    return word_inhibition_matrix

def get_pred_dict(pm, lexicon):

    output_word_pred_map = f"data/processed/prediction_map_{pm.stim_name}_{pm.prediction_flag}_{pm.task_to_run}_{pm.language}.json"
    if pm.prediction_flag in ['gpt2', 'llama']:
        output_word_pred_map = output_word_pred_map.replace('.json', f'_topk{pm.topk}.json')

    # AL: in case pred file needs to be created from original files
    if not os.path.exists(output_word_pred_map):
        create_pred_file(pm, output_word_pred_map, lexicon)

    with open(output_word_pred_map, "r") as f:
        word_pred_dict = json.load(f)

    return word_pred_dict

# def write_out_simulation_data_specific(simulation_data, outfile_sim_data):

    # simulation_results = defaultdict(list)

    # for trial_data in simulation_data:  # Iterate through each trial's data
    #     for key, values in trial_data.items():
    #         if isinstance(values, list):
    #             simulation_results[key].extend(values)
    #         else:
    #             simulation_results[key].append(values)

    # simulation_results_df = pd.DataFrame.from_dict(simulation_results)
    # simulation_results_df.to_csv(outfile_sim_data, sep='\t', index=False)

def write_out_simulation_data_specific(simulation_data, outfile_sim_data):

    simulation_results = defaultdict(list)

    for sim_index, texts_simulations in simulation_data.items():
        for text_index, text in texts_simulations.items():
            for fix_counter, fix_info in text.items():
                simulation_results['fixation_counter'].append(fix_counter)
                simulation_results['text_id'].append(text_index)
                simulation_results['simulation_id'].append(sim_index)
                for info_name, info_value in fix_info.items():
                    simulation_results[info_name].append(info_value)

    simulation_results_df = pd.DataFrame.from_dict(simulation_results)
    simulation_results_df.to_csv(outfile_sim_data, sep='\t', index=False)