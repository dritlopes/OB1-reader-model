import logging
import numpy as np
import pickle
from collections import defaultdict
import math
from tqdm import tqdm
from time import sleep
from reading_helper_functions_specific_tasks import string_to_bigrams_and_locations
from simulate_experiment_specific_tasks_helper_functions import build_lexicon, compute_overlap_and_inhibition_matrices, compute_word_threshold, display_overlap_and_inhibition_matrices, generate_lexicon_mappings, identify_realwords, process_affixes, process_and_filter_words, process_tokens
from utils_specific_tasks import calculate_unit_activations, get_pred_dict, get_word_freq, get_pred_values, inter_word_inhibition, map_stimuli_words_to_lexicon_indices, pre_process_string, set_up_inhibition_matrix
from reading_components_specific_tasks import affix_modelling_underscores, lexicon_bigrams
from reading_components import compute_stimulus, compute_eye_position, compute_words_input, update_word_activity, match_active_words_to_input_slots, compute_next_attention_position, compute_next_eye_position
from reading_helper_functions import get_threshold, string_to_open_ngrams, define_slot_matching_order, sample_from_norm_distribution, find_word_edges, compute_entropy
from word_recognition_specific_tasks_helper_functions import all_ngrams_and_bigrams, calculate_word_input, check_recognition_status, compute_trial_metrics, determine_correctness, get_blankscreen_stimulus, handle_trial_task_specifics, initialize_trial, initialize_trial_data, match_words_in_slots, reset_recognition_variables, set_stimulus, slot_matching_order

np.set_printoptions(threshold=np.inf) ### to print large np arrays
logger = logging.getLogger(__name__)


def reading(pm,tokens,word_overlap_matrix,lexicon_word_ngrams,lexicon_word_index,lexicon,pred_dict,freq_values,verbose=True):

    all_data = {}
    # set to true when end of text is reached
    end_of_text = False
    # the element of fixation in the text. It goes backwards in case of regression
    fixation = 0
    # +1 with every next fixation
    fixation_counter = 0
    # initialize eye position
    eye_position = None
    # initialise attention window size
    attend_width = pm.attend_width
    # total number of tokens in input
    total_n_words = len(tokens)
    # word activity for word in lexicon
    lexicon_word_activity = np.zeros((len(lexicon)), dtype=float)
    # keep track whether a prediction was made at a given position (for pred-att mechanism)
    predicted = False
    # # keep track of the highest prediction value for each position in the text
    # highest_predictions = dict()
    # history of regressions, set to true at a certain position in the text when a regression is performed to that word
    regression_flag = np.zeros(total_n_words, dtype=bool)
    # recognized word at position, which word received the highest activation in each position
    recognized_word_at_position = np.empty(total_n_words, dtype=object)
    # the amount of cycles needed for each word in text to be recognized
    recognized_word_at_cycle = np.zeros(total_n_words, dtype=int)
    recognized_word_at_cycle.fill(-1)
    # info on saccade for each eye movement
    saccade_info = {'saccade_type': None,  # regression, forward, refixation or wordskip
                    'saccade_distance': 0,  # distance between current eye position and eye previous position
                    'saccade_error': 0,  # saccade noise to include overshooting
                    'saccade_cause': 0}  # for wordskip and refixation, extra info on cause of saccade
    saccade_symbols = {'forward': ">->->->->->->->->->->->-",
                       'wordskip': ">>>>>>>>>>>>>>>>>>>>>>>>",
                       'refixation': '------------------------',
                       'regression': '<-<-<-<-<-<-<-<-<-<-<-<-'}

    # mapping between token position in text and lexicon index
    tokens_to_lexicon_indices = np.zeros((total_n_words), dtype=int)
    for i, word in enumerate(tokens):
        tokens_to_lexicon_indices[i] = lexicon.index(word)

    # if pm.prediction_flag:
    #     for i in enumerate(tokens):
    #         if i in pred_dict.keys():
    #             highest_predictions[i] = pred_dict[i].values().tolist()[0]

    if pm.prediction_flag:
        entropy_values = compute_entropy(pred_dict)

    # ---------------------- Start looping through fixations ---------------------
    while not end_of_text:

        if verbose:
            print(f'---Fixation {fixation_counter} at position {fixation}---')
        logger.info(f'---Fixation {fixation_counter} at position {fixation}---')

        # make sure that fixation does not go over the end of the text. Needed for continuous reading
        fixation = min(fixation, len(tokens) - 1)

        fixation_data = {"foveal_word": tokens[fixation],
                         'foveal_word_index': fixation,
                         'foveal_word_frequency': freq_values[tokens[fixation]] if tokens[fixation] in freq_values.keys() else 0,
                         'foveal_word_predictability': pred_dict[str(fixation)][tokens[fixation]] if pred_dict and str(fixation) in pred_dict.keys() and tokens[fixation] in pred_dict[str(fixation)].keys() else 0,
                         'foveal_word_length': len(tokens[fixation]),
                         'foveal_word_threshold': pm.max_threshold,
                         'stimulus': None,
                         'trial_words': tokens,
                         'eye_position': None,
                         'attentional_width': attend_width,
                         'saccade_type': saccade_info['saccade_type'],
                         'saccade_error': saccade_info['saccade_error'],
                         'saccade_distance': saccade_info['saccade_distance'],
                         'saccade_cause': saccade_info['saccade_cause'],
                         'recognition_cycle': None,
                         'cycle_of_recognition': recognized_word_at_cycle,
                         'fixation_duration': None,
                         'recognized_word_at_foveal_position': None,
                         'recognized_words': None
                         }

        # ---------------------- Define the stimulus and eye position ---------------------
        stimulus, stimulus_position, fixated_position_in_stimulus = compute_stimulus(fixation, tokens)
        eye_position = compute_eye_position(stimulus, fixated_position_in_stimulus, eye_position)
        fixation_data['stimulus'] = stimulus
        fixation_data['eye_position'] = eye_position
        if verbose: print(f"Stimulus: {stimulus}\nEye position: {eye_position}")
        logger.info(f"Stimulus: {stimulus}\nEye position: {eye_position}")

        # ---------------------- Update attention width ---------------------
        # update attention width according to whether there was a regression in the last fixation,
        # i.e. this fixation location is a result of regressio
        if fixation_data['saccade_type'] == 'regression':
            # set regression flag to know that a regression has been realized towards this position
            regression_flag[fixation] = True
            # narrow attention width by 2 letters in the case of regressions
            attend_width = max(attend_width - 1.0, pm.min_attend_width)
        else:
            # widen atention by 0.5 letters in forward saccades
            attend_width = min(attend_width + 0.5, pm.max_attend_width)

        # ---------------------- Define order of slot-matching ---------------------
        # define order to match activated words to slots in the stimulus
        # NV: the order list should reset when stimulus changes or with the first stimulus
        order_match_check = define_slot_matching_order(len(stimulus.split()), fixated_position_in_stimulus,
                                                       attend_width)
        #print(f'Order for slot-matching: {order_match_check}')

        # ---------------------- Start processing of stimulus ---------------------
        #print('Entering cycle loops to define word activity...')
        if verbose:
            print(f"fix on: {tokens[fixation]}  attent. width: {attend_width}")
            if tokens[fixation] in freq_values.keys():
                print(f'   fixwrd freq. {freq_values[tokens[fixation]]}')
        logger.info(f"fix on: {tokens[fixation]}  attent. width: {attend_width}")
        if tokens[fixation] in freq_values.keys():
            logger.info(f'   fixwrd freq. {freq_values[tokens[fixation]]}')
        # str(round(lexicon_thresholds[tokens_to_lexicon_indices[fixation]],3))
        shift = False
        n_cycles = 0
        n_cycles_since_attent_shift = 0
        attention_position = eye_position
        # stimulus position in which recognition is achieved during current fixation
        recognition_in_stimulus = []
        # define index of letters at the words edges.
        word_edges = find_word_edges(stimulus)

        # ---------------------- Define word excitatory input ---------------------
        # compute word input using ngram excitation and inhibition
        n_ngrams, total_ngram_activity, all_ngrams, word_input = compute_words_input(stimulus, lexicon_word_ngrams, eye_position, attention_position, attend_width, pm, freq_values)
        # fixation_data['n_ngrams'] = n_ngrams
        # fixation_data['total_ngram_activity'] = total_ngram_activity
        if verbose:
            print(f"  input to fixwrd at first cycle: {round(word_input[tokens_to_lexicon_indices[fixation]], 3)}")
        logger.info(f"  input to fixwrd at first cycle: {round(word_input[tokens_to_lexicon_indices[fixation]], 3)}")

        # Counter n_cycles_since_attent_shift is 0 until attention shift (saccade program initiation),
        # then starts counting to 5 (because a saccade program takes 5 cycles, or 125ms.)
        while n_cycles_since_attent_shift < 5:

            # AL: if a word has been recognized during this fixation and attention has not shifted yet,
            # recompute ngram activation such that ngram activation from matched words is removed until attention shifts
            if recognition_in_stimulus and attention_position != None: # and not shift

                n_ngrams, total_ngram_activity, all_ngrams, word_input = compute_words_input(stimulus,
                                                                                             lexicon_word_ngrams,
                                                                                             eye_position,
                                                                                             attention_position,
                                                                                             attend_width, pm,
                                                                                             freq_values,
                                                                                             recognition_in_stimulus,
                                                                                             tokens,
                                                                                             recognized_word_at_cycle,
                                                                                             n_cycles)

            # ---------------------- Update word activity per cycle ---------------------
            # Update word act with word inhibition (input remains same, so does not have to be updated)
            lexicon_word_activity, lexicon_word_inhibition = update_word_activity(lexicon_word_activity,
                                                                                  word_overlap_matrix,
                                                                                  pm, word_input,
                                                                                  len(lexicon))


            #print("input these:", word_input[lexicon.index('these')])
            #print("inhib these:", lexicon_word_inhibition[lexicon.index('these')])
            #print("activ these:", lexicon_word_activity[lexicon.index('these')])

            # update cycle info
            foveal_word_index = lexicon_word_index[tokens[fixation]]
            foveal_word_activity = lexicon_word_activity[foveal_word_index]
            if verbose:
               print(f'CYCLE {n_cycles}    activ @fix {round(foveal_word_activity, 3)} inhib  #@fix {round(lexicon_word_inhibition[foveal_word_index], 6)}')
            logger.info(f'CYCLE {n_cycles}    activ @fix {round(foveal_word_activity, 3)} inhib  #@fix {round(lexicon_word_inhibition[foveal_word_index], 6)}')

            # ---------------------- Match words in lexicon to slots in input ---------------------
            # word recognition, by checking matching active wrds to slots
            recognized_word_at_position, lexicon_word_activity, recognition_in_stimulus = \
                match_active_words_to_input_slots(order_match_check,
                                                  stimulus,
                                                  recognized_word_at_position,
                                                  lexicon_word_activity,
                                                  lexicon,
                                                  pm.min_activity,
                                                  stimulus_position,
                                                  pm.word_length_similarity_constant,
                                                  recognition_in_stimulus,
                                                  pm.max_threshold,
                                                  verbose)

            if fixation < total_n_words-1 and pm.prediction_flag:
                # gradually pre-activate words in stimulus (pred weighted by pred of previous word)
                for position in range(fixation+1, fixation+len(stimulus.split(' '))):
                    if position > 0 and position < len(tokens):
                        if verbose: print(f'POSITION {position}')
                        logger.info(f'POSITION {position}')
                        if not recognized_word_at_position[position]:
                            lexicon_word_activity, predicted = activate_predicted_upcoming_word(position,
                                                                                                tokens[position],
                                                                                                fixation,
                                                                                                lexicon_word_activity,
                                                                                                lexicon,
                                                                                                pred_dict,
                                                                                                pm.pred_weight,
                                                                                                recognized_word_at_position,
                                                                                                predicted,
                                                                                                entropy_values,
                                                                                                verbose)

            # ---------------------- Make saccade decisions ---------------------
            # word selection and attention shift
            if not shift:
                # MM: on every cycle, take sample (called shift_start) out of normal distrib.
                # If cycle since fixstart > sample, make attentshift. This produces approx ex-gauss SRT
                if recognized_word_at_position[fixation]:
                    # MM: if word recog, then faster switch (norm. distrib. with <mu) than if not recog.
                    shift_start = sample_from_norm_distribution(pm.mu, pm.sigma, pm.recog_speeding, recognized=True)
                else:
                    shift_start = sample_from_norm_distribution(pm.mu, pm.sigma, pm.recog_speeding, recognized=False)
                # shift attention (& plan saccade in 125 ms) if n_cycles is higher than random threshold shift_start
                if n_cycles >= shift_start:
                    shift = True
                    # AL: re-set saccade info from current fixation to update it with next fixation
                    saccade_info = {'saccade_type': None,
                                    'saccade_distance': 0,
                                    'saccade_error': 0,
                                    'saccade_cause': 0}
                    attention_position = compute_next_attention_position(all_data,
                                                                        tokens,
                                                                        fixation,
                                                                        word_edges,
                                                                        fixated_position_in_stimulus,
                                                                        regression_flag,
                                                                        recognized_word_at_position,
                                                                        lexicon_word_activity,
                                                                        eye_position,
                                                                        fixation_counter,
                                                                        attention_position,
                                                                        attend_width,
                                                                        foveal_word_index,
                                                                        pm,
                                                                        verbose)

                    predicted = False

                    if verbose:
                        print(f'attentpos {attention_position}')
                    logger.info(f'attentpos {attention_position}')

                    # AL: attention position is None if at the end of the text and saccade is not refixation nor regression, so do not compute new words input
                    if attention_position != None:
                        # AL: recompute word input, using ngram excitation and inhibition, because attentshift changes bigram input
                        n_ngrams, total_ngram_activity, all_ngrams, word_input = compute_words_input(stimulus,
                                                                                                     lexicon_word_ngrams,
                                                                                                     eye_position,
                                                                                                     attention_position,
                                                                                                     attend_width,
                                                                                                     pm,
                                                                                                     freq_values,
                                                                                                     recognition_in_stimulus,
                                                                                                     tokens,
                                                                                                     recognized_word_at_cycle,
                                                                                                     n_cycles)
                        attention_position = np.round(attention_position)

                    if verbose: print(f"  input after attentshift: {round(word_input[tokens_to_lexicon_indices[fixation]], 3)}")
                    logger.info(f"  input after attentshift: {round(word_input[tokens_to_lexicon_indices[fixation]], 3)}")

            if shift:
                n_cycles_since_attent_shift += 1 # ...count cycles since attention shift

            for i in stimulus_position:
                if recognized_word_at_position[i] and recognized_word_at_cycle[i] == -1: # recognized_word_at_position[fixation]
                    # MM: here the time to recognize the word gets stored
                    recognized_word_at_cycle[i] = n_cycles
                    if i == fixation:
                        fixation_data['recognition_cycle'] = recognized_word_at_cycle[fixation]
                        fixation_data['cycle_of_recognition'] = recognized_word_at_cycle

            n_cycles += 1

        # out of cycle loop. After last cycle, compute fixation duration and add final values for fixated word before shift is made
        fixation_duration = n_cycles * pm.cycle_size
        fixation_data['fixation_duration'] = fixation_duration
        fixation_data['recognized_words'] = recognized_word_at_position

        if verbose:
            print(f"Fixation duration: {fixation_duration} ms.")
        logger.info(f"Fixation duration: {fixation_duration} ms.")

        if recognized_word_at_position[fixation]:
            fixation_data['recognized_word_at_foveal_position'] = recognized_word_at_position[fixation]
            if verbose:
                if recognized_word_at_position[fixation] == tokens[fixation]:
                    if verbose:
                        print("Correct word recognized at fixation!")
                    logger.info("Correct word recognized at fixation!")
                else:
                    if verbose:
                        print(f"Wrong word recognized at fixation! (Recognized: {recognized_word_at_position[fixation]})")
                    logger.info(f"Wrong word recognized at fixation! (Recognized: {recognized_word_at_position[fixation]})")
        else:
            fixation_data['recognized_word_at_foveal_position'] = ""
            if verbose:
                print("No word was recognized at fixation position")
                print(f"Word with highest activation: {lexicon[np.argmax(lexicon_word_activity)]}")
            logger.info("No word was recognized at fixation position")
            logger.info(f"Word with highest activation: {lexicon[np.argmax(lexicon_word_activity)]}")

        # add fixation dict to list of dicts
        all_data[fixation_counter] = fixation_data
        if verbose:
            print(recognized_word_at_position) #fixation_data)
        logger.info(recognized_word_at_position)

        fixation_counter += 1

        # compute next eye position and thus next fixation
        if verbose:
            print(f'att pos right before computing next eye position: {attention_position}')
        logger.info(f'att pos right before computing next eye position: {attention_position}')

        if attention_position != None:

            fixation, eye_position, saccade_info = compute_next_eye_position(pm, attention_position, eye_position, fixation, fixated_position_in_stimulus, word_edges, saccade_info, verbose)

            # AL: Update saccade cause for next fixation
            if saccade_info['saccade_type'] == 'wordskip':
                if regression_flag[fixation]:
                    saccade_info['saccade_cause'] = 2  # AL: bcs n resulted from regression and n + 1 has been recognized
                else:
                    saccade_info['saccade_cause'] = 1  # AL: bcs n + 2 has highest attwght (letter excitation)

            elif saccade_info['saccade_type'] == 'refixation':
                if not recognized_word_at_position[fixation]:
                    saccade_info['saccade_cause'] = 1  # AL: bcs fixated word has not been recognized
                else:
                    saccade_info['saccade_cause'] = 2  # AL: bcs right of fixated word has highest attwght (letter excitation)

        # Check if end of text is reached AL: if fixation on last word and next saccade not refixation nor regression
        if fixation_data['foveal_word_index'] == total_n_words - 1 and saccade_info['saccade_type'] not in ['refixation', 'regression']:
            end_of_text = True
            continue
        elif attention_position == None:
            end_of_text = True
            continue
        else:
            if saccade_info['saccade_type']:
                if verbose:
                    print(saccade_symbols[saccade_info['saccade_type']])
                logger.info(saccade_symbols[saccade_info['saccade_type']])

    return all_data

def word_recognition(pm,word_inhibition_matrix,lexicon_word_ngrams,lexicon_word_index,lexicon_thresholds_dict,lexicon,word_frequencies):

    # information computed for each fixation
    all_data = {}
    # data frame with stimulus info
    stim_df = pm.stim
    # list of stimuli
    stim = pm.stim_all
    # initialise attention window size
    attend_width = pm.attend_width
    # word activity for each word in lexicon
    lexicon_word_activity = np.zeros((len(lexicon)), dtype=float)
    # recognition threshold for each word in lexicon
    lexicon_thresholds = np.zeros((len(lexicon)), dtype=float)
    # update lexicon_thresholds with frequencies if dict filled in (if frequency flag)
    if lexicon_thresholds_dict != {}:
        for i, word in enumerate(lexicon):
            lexicon_thresholds[i] = lexicon_thresholds_dict[word]

    for trial in range(0, len(pm.stim_all)):

        trial_data = defaultdict(list)

        # ---------------------- Define the trial stimuli ---------------------
        stimulus = stim[trial]
        fixated_position_in_stim = math.floor(len(stimulus.split(' '))/2)
        eye_position = np.round(len(stimulus) // 2)
        target = stim_df['target'][trial]
        condition = stim_df['condition'][trial]
        trial_data['stimulus'] = stimulus
        trial_data['eye position'] = eye_position
        trial_data['target'] = target
        trial_data['condition'] = condition
        if pm.is_priming_task:
            prime = stim_df['prime'][trial]
            trial_data['prime'] = prime

        trial_data['target threshold'] = lexicon_thresholds[lexicon_word_index[target]]
        trial_data['target frequency'] = word_frequencies[target]
        # if pred_values:
        #     trial_data['target predictability'] = pred_values[stimulus.split().index(target)]

        # ---------------------- Start processing stimuli ---------------------
        n_cycles = 0
        recog_cycle = 0
        attention_position = eye_position
        stimuli = []
        recognized, false_guess = False,False
        # init activity matrix with min activity. Assumption that each trial is independent.
        lexicon_word_activity[lexicon_word_activity < pm.min_activity] = pm.min_activity

        # keep processing stimuli as long as trial lasts
        while n_cycles < pm.totalcycles:

            # stimulus may change within a trial to blankscreen, prime
            if n_cycles < pm.blankscreen_cycles_begin or n_cycles > pm.totalcycles - pm.blankscreen_cycles_end:
                stimulus = get_blankscreen_stimulus(pm.blankscreen_type)
                stimuli.append(stimulus)
                n_cycles += 1
                continue
            elif pm.is_priming_task and n_cycles < (pm.blankscreen_cycles_begin+pm.ncyclesprime):
                stimulus = prime
            else:
                stimulus = stim[trial]
            stimuli.append(stimulus)

            # keep track of which words have been recognized in the stimulus
            # create array if first stimulus or stimulus has changed within trial
            if (len(stimuli) <= 1) or (stimuli[-2] != stimuli[-1]):
                stimulus_matched_slots = np.empty(len(stimulus.split()),dtype=object)

            # define order for slot matching. Computed within cycle loop bcs stimulus may change within trial
            order_match_check = define_slot_matching_order(len(stimulus.split()),fixated_position_in_stim,attend_width)

            # compute word excitatory input given stimulus
            n_ngrams, total_ngram_activity, all_ngrams, word_input = compute_words_input(stimulus,
                                                                                         lexicon_word_ngrams,
                                                                                         eye_position,
                                                                                         attention_position,
                                                                                         attend_width,
                                                                                         pm)
            trial_data['ngram act per cycle'].append(total_ngram_activity)
            trial_data['n of ngrams per cycle'].append(len(all_ngrams))

            # update word activity using word-to-word inhibition and decay
            lexicon_word_activity, lexicon_word_inhibition = update_word_activity(lexicon_word_activity,
                                                                                  word_inhibition_matrix,
                                                                                  pm,
                                                                                  word_input,
                                                                                  len(lexicon))
            trial_data['target act per cycle'].append(lexicon_word_activity[lexicon_word_index[target]])
            trial_data['lexicon act per cycle'].append(sum(lexicon_word_activity))
            trial_data['stimulus act per cycle'].append(sum([lexicon_word_activity[lexicon_word_index[word]] for word in stimulus.split() if word in lexicon_word_index.keys()]))

            # word recognition, by checking matching active wrds to slots
            stimulus_matched_slots, lexicon_word_activity = \
                match_active_words_to_input_slots(order_match_check,
                                                  stimulus,
                                                  stimulus_matched_slots,
                                                  lexicon_thresholds,
                                                  lexicon_word_activity,
                                                  lexicon,
                                                  pm.min_activity,
                                                  None,
                                                  pm.word_length_similarity_constant)

            # register cycle of recognition at target position
            if target in stimulus.split() and stimulus_matched_slots[fixated_position_in_stim]:
                recog_cycle = n_cycles
                if stimulus_matched_slots[fixated_position_in_stim] == target:
                    recognized = True
                else:
                    false_guess = True

            # NV: if the design of the task considers the first recognized word in the target slot to be the final response, stop the trial when this happens
            if pm.trial_ends_on_key_press and (recognized == True or false_guess == True):
                break

            n_cycles += 1

        # compute reaction time
        reaction_time = ((recog_cycle + 1 - pm.blankscreen_cycles_begin) * pm.cycle_size) + 300
        print("reaction time: " + str(reaction_time) + " ms")
        print("end of trial")
        print("----------------")
        print("\n")

        trial_data['attend width'] = attend_width
        trial_data['reaction time'] = reaction_time
        trial_data['matched slots'] = stimulus_matched_slots
        trial_data['target recognized'] = recognized
        trial_data['false guess'] = false_guess

        all_data[trial] = trial_data

        for key,value in trial_data.items():
            print(key, ': ',value)
        if trial > 1: exit()

    return all_data

def simulate_experiment(pm):

    verbose = pm.print_process

    if verbose: print('\nPreparing simulation(s)...')
    logger.info('\nPreparing simulation(s)...')
    tokens = [token for stimulus in pm.stim_all for token in stimulus.split(' ') if token != '']

    if pm.is_priming_task:
        tokens.extend([token for stimulus in list(pm.stim["prime"]) for token in stimulus.split(' ')])

    tokens = [pre_process_string(token) for token in tokens]
    # remove empty strings which were once punctuations
    tokens = [token for token in tokens if token != '']
    word_frequencies = get_word_freq(pm, set(tokens), n_high_freq_words=500)
    max_frequency = max(word_frequencies.values())
    lexicon = list(set(tokens) | set(word_frequencies.keys()))
    lexicon = [pre_process_string(word) for word in lexicon]

    printwords=False
    if printwords:
        print('LONG or SHORT WORDS')
        logger.info('LONG or SHORT WORDS')
        for token in set(tokens):
            if len(token) > 10 or len(token) == 1:
                print(f'{token}, len: {len(token)}, count: {tokens.count(token)}')
        if verbose: print('\nLOW-FREQUENCY WORDS')
        logger.info('\nLOW-FREQUENCY WORDS')
        for token in set(tokens):
            if token in word_frequencies.keys():
                if word_frequencies[token] <= 1.5:
                    if verbose: print(f'{token}, freq: {word_frequencies[token]}, count: {tokens.count(token)}')
                    logger.info(f'{token}, freq: {word_frequencies[token]}, count: {tokens.count(token)}')

    # write out lexicon for consulting purposes
    lexicon_filename = 'data\processed\lexicon.pkl'
    with open(lexicon_filename, "wb") as f:
        pickle.dump(lexicon, f)

    if verbose:
        print('\nSetting word recognition thresholds...')
    logger.info('\nSetting word recognition thresholds...')
    # define word recognition thresholds
    word_thresh_dict = {}
    for word in lexicon:
        if pm.frequency_flag:
            word_thresh_dict[word] = get_threshold(word,
                                                   word_frequencies,
                                                   max_frequency,
                                                   pm.freq_weight,
                                                   pm.max_threshold)

    # lexicon indices for each word of text
    total_n_words = len(tokens)
    tokens_to_lexicon_indices = np.zeros((total_n_words), dtype=int)
    for i, word in enumerate(tokens):
        tokens_to_lexicon_indices[i] = lexicon.index(word)

    if verbose: print('Finding ngrams from lexicon...')
    logger.info('Finding ngrams from lexicon...')
    # lexicon bigram dict
    lexicon_word_ngrams = {}
    lexicon_word_index = {}
    for i, word in enumerate(lexicon):
        # AL: weights and locations are not used for lexicon, only the ngrams of the words in the lexicon for comparing them later with the ngrams activated in stimulus.
        all_word_ngrams, weights, locations = string_to_open_ngrams(word,pm.bigram_gap)
        lexicon_word_ngrams[word] = all_word_ngrams
        lexicon_word_index[word] = i

    if verbose: print('Computing word-to-word inhibition matrix...')
    logger.info('Computing word-to-word inhibition matrix...')
    # set up word-to-word inhibition matrix
    word_inhibition_matrix = set_up_inhibition_matrix(pm, lexicon, lexicon_word_ngrams)
    if verbose: print("Inhibition grid ready.")
    logger.info("Inhibition grid ready.")
    # print("Inhib from the to these:", word_inhibition_matrix[lexicon.index('these'),lexicon.index('the')])

    # recognition threshold for each word in lexicon
    lexicon_thresholds = np.zeros((len(lexicon)), dtype=float)
    # initialize thresholds with values based frequency, if dict has been filled (if frequency flag)
    if word_thresh_dict != {}:
        for i, word in enumerate(lexicon):
            lexicon_thresholds[i] = word_thresh_dict[word]

    if verbose:
        print("")
        print("BEGIN SIMULATION(S)")
        print("")
    logger.info("BEGIN SIMULATION(S)")

    # how many trials/texts from corpus/data should be used
    if pm.n_trials == 0 or pm.n_trials > len(pm.stim_all):
        pm.n_trials = len(pm.stim_all)

    # read text/trials
    all_data = defaultdict()

    for sim_number in range(pm.number_of_simulations):

        if verbose: print(f"SIMULATION {sim_number}")
        logger.info(f"SIMULATION {sim_number}")

        if pm.task_to_run == 'continuous_reading':

            texts_simulations = defaultdict()

            # AL: if language model, generate predictions
            word_predictions = get_pred_dict(pm, lexicon)

            # initiate progress bar
            pbar = tqdm(total=pm.n_trials)

            for i, text in enumerate(pm.stim_all[:pm.n_trials]):

                text_tokens = [pre_process_string(token) for token in text.split()]

                predictions_in_text = None
                if word_predictions:
                    predictions_in_text = word_predictions[str(i)]

                text_data = reading(pm,
                                    text_tokens,
                                    word_inhibition_matrix,
                                    lexicon_word_ngrams,
                                    lexicon_word_index,
                                    lexicon,
                                    predictions_in_text,
                                    word_frequencies,
                                    verbose=verbose)

                texts_simulations[i] = text_data

                # progress bar update
                sleep(0.1)
                pbar.update(1)

            all_data[sim_number] = texts_simulations
            # close progress bar
            pbar.close()

        else:
            all_data = word_recognition(pm,
                                        word_inhibition_matrix,
                                        lexicon_word_ngrams,
                                        lexicon_word_index,
                                        word_thresh_dict,
                                        lexicon,
                                        word_frequencies)

    if verbose: print(f'THE END')
    logger.info(f'THE END')

    return all_data

def word_recognition_specific_tasks(pm, word_overlap_matrix, tokens, affixes, lexicon_size, lexicon_normalized_word_inhibition, lexicon_thresholds_np, word_inhibition_matrix, lexicon_word_bigrams, bigram_monogram_total, lexicon_indices_dict,word_thresh_dict,lexicon,pred_values,word_frequencies,):
    """
    The function simulates the process of word recognition based on the provided parameters and stimuli.
    Steps:
    1. Initialization: Sets up necessary variables such as the lexicon, word activity, and thresholds. 
        Additionally, it prepares the data structures to store the results of the simulation.
    2. Trial Iteration: Iterates through each trial (e.g., each word or stimulus). 
        For each trial, the function simulates the recognition process through several cycles, 
        adjusting word activity levels based on various factors like overlap with other words, 
        inhibition from other active words.
    3. Recognition Decision: At the end of the cycles for each trial,
        the function determines if the word was recognised, misrecognised, 
        or not recognised based on the activity levels of words in the lexicon.
    4. Data Recording: For each trial, various metrics and results such as word activity levels, 
        recognized words, reaction times, and more are recorded.

    Parameters:
    - matrices and vectors that represent the lexicon, word overlap, word inhibition, word thresholds, and other metrics.
    - The `pm` parameter, which provides parameters.

    Returns:
    - A structured data object (`all_data`) that contains the results of the simulation for each trial, 
        including metrics like recognized words, reaction times, word activity levels.
    """
    
    """
    Initialise all the variables
    """
    # Initialise a list to store data for all trials
    all_data = []
    # Extract the stimuli data from the parameters
    stim_df = pm.stim
    # Get all stimuli, including non-words
    stim = pm.stim_all
    # If there's a priming task, extract the prime words
    if pm.is_priming_task:
        primes = pm.stim['prime']
    
    # Set the attention width from the provided parameters
    attend_width = pm.attend_width
    lexicon_word_activity = np.zeros((len(lexicon)), dtype=float)
    lexicon_thresholds = lexicon_thresholds_np
    # List to keep track of words that weren't recognized
    unrecognized_words = []

    # numpy arrays for various lexicon based metrics
    lexicon_word_activity_np = np.zeros((lexicon_size), dtype=float) # Activity level for each word in lexicon
    lexicon_word_inhibition_np = np.zeros((lexicon_size), dtype=float) # Inhibition level for each word in lexicon
    lexicon_word_inhibition_np2 = np.zeros((lexicon_size), dtype=float) # Secondary inhibition level (if used)
    lexicon_activewords_np = np.zeros((lexicon_size), dtype=int) # List of words that are active in the lexicon
    word_input_np = np.zeros((lexicon_size), dtype=float) # Input to each word in the lexicon

    # List to track all words that were searched during the process
    all_words_searched = []
    
    """
    Loop through each trial
    """
    for trial in range(0, len(stim)):
        print(f"----- Trial {trial + 1} -----")

        # Initialise variables for each trial
        (EyePosition, AttentionPosition, 
         lexicon_word_inhibition_np, lexicon_total_input_np, lexicon_word_activity_change, lexicon_word_activity_np, 
         crt_word_activity_np, stimuli) = initialize_trial(lexicon_size, pm, stim, trial)
    
        # Extract and set stimulus-related information for the current trial, 
            # such as the padded stimulus, fixation point, and prime
        (stimulus, stimulus_padded, 
         fixated_position_in_stim, eye_position, attention_position, 
         prime, prime_padded) = set_stimulus(trial, stim, pm, primes)
    
         # Initialise a data dictionary for the current trial to store results and metrics 
        trial_data = initialize_trial_data(trial, pm, stimulus, prime, stim_df, EyePosition, AttentionPosition)
        all_data.append(trial_data)
    
        # Based on the type of task being run, extract and set the target word for the current trial
        target = handle_trial_task_specifics(trial, pm, stimulus, stim_df, prime, all_data)

        # Reset recognition-related variables for the new trial,
            # such as the current cycle number, recognition flags
        (cycle_for_RT, cur_cycle, 
         recognized, falseguess, grammatical, identif, 
         POSrecognition) = reset_recognition_variables()

        """
        Go through cycles for each trial
        """
        # Loop through each cycle for the current trial
        while cur_cycle < pm.totalcycles:
            print(f"--- Cycle {cur_cycle + 1} ---")
            # Set the stimulus for the current cycle, based on conditions for blank screen cycles and priming task
            if cur_cycle < pm.blankscreen_cycles_begin or cur_cycle > pm.totalcycles-pm.blankscreen_cycles_end:
                stimulus, stimulus_padded = get_blankscreen_stimulus(pm.blankscreen_type)
            elif pm.is_priming_task and cur_cycle < (pm.blankscreen_cycles_begin+pm.ncyclesprime):
                stimulus = prime
                stimulus_padded = prime_padded
            else:
                stimulus = stim[trial]
                stimulus_padded = stimulus_padded

            # Count the number of words in the stimulus
            n_words_in_stim = len(stimulus.split())

            # Determine the order for slot matching
            order_match_check = slot_matching_order(stimulus)
            stimuli.append(stimulus)
            
            # checks whether the last two items in a list are different or whether the list has less than or equal to 1 item.
                # If yes, creates an empty numpy array of the same length as the number of words in the stimulus
            if (len(stimuli) <= 1) or (stimuli[-2] != stimuli[-1]):
                stimulus_matched_slots = [""] * n_words_in_stim
                for slot_to_check in range(0, len(stimulus.split())):
                    POSrecognition[slot_to_check] = ''

            # Convert the stimulus into bigrams and their respective locations
            (all_ngrams,bigrams_to_locations) = string_to_bigrams_and_locations(stimulus_padded, is_prefix=False, is_suffix=False)
        
             # compute word excitatory input given stimulus
            all_bigrams, all_monograms, all_bigrams_set = all_ngrams_and_bigrams(all_ngrams)

            # Reset word input and inhibition values
            word_input_np.fill(0.0)
            lexicon_word_inhibition_np.fill(0.0)
            lexicon_word_inhibition_np2.fill(0.0)
            lexicon_activewords_np.fill(False)

            # Calculate the activity of each ngram unit based on eye and attention positions
            unit_activations = calculate_unit_activations(all_ngrams, bigrams_to_locations, eye_position, AttentionPosition, pm, attend_width)
            all_data[trial]['bigram activity per cycle'].append(sum(unit_activations.values()))
            all_data[trial]['ngrams'].append(len(all_ngrams))

            # Compute the input for each word in the lexicon based on the active ngram units
            word_input_np = calculate_word_input(unit_activations, pm, all_ngrams, lexicon, lexicon_word_bigrams, all_bigrams_set, all_monograms, bigram_monogram_total)
            

            # Calculate the activity of each word in the lexicon considering the word's input and inter-word inhibition
            lexicon_word_activity_np, lexicon_total_input_np, squared_activity, lexicon_word_inhibition_np, overlap_select, lexicon_select = inter_word_inhibition(
                lexicon_word_activity_np, 
                lexicon_total_input_np, 
                lexicon_activewords_np,  
                word_input_np, 
                pm, 
                word_overlap_matrix,   
                lexicon_normalized_word_inhibition, 
                bigram_monogram_total,        
                lexicon, 
                target
            )

            # Track the activity and inhibition of the target word
            target_lexicon_index = [lexicon_indices_dict[element] for idx, element in enumerate(lexicon) if element == '_'+target+'_']
            all_data[trial]['target_inhib'].append(lexicon_word_inhibition_np[target_lexicon_index])
            crt_word_activity_np = lexicon_word_activity_np[target_lexicon_index]
            all_data[trial]['target activity per cycle'].append(crt_word_activity_np)

            # Calculate the stimulus activity and total lexicon activity
            stim_activity = sum([lexicon_word_activity_np[lexicon_indices_dict['_'+word+'_']]for word in stimulus.split() if '_'+word+'_' in lexicon])
            total_activity = sum(lexicon_word_activity_np)
            sum_of_squared_activity = sum(squared_activity)

            # Store total lexicon activity and squared activity
            all_data[trial]['lexicon activity per cycle'].append(total_activity)
            all_data[trial]['lexicon activity squared'].append(sum_of_squared_activity)

            # Determine which words in the lexicon have activity above their respective thresholds
            above_thresh_lexicon_np = np.where(lexicon_word_activity_np > lexicon_thresholds_np, 1, 0)
            # Store the current cycle number
            all_data[trial]['cycle'].append(cur_cycle)

            # # Identify and store lexicon words and their indices that are above threshold
            sorted_indices = np.argsort(lexicon_word_activity_np)[::-1]
            above_threshold_indices = sorted_indices[lexicon_word_activity_np[sorted_indices] > lexicon_thresholds_np[sorted_indices]]
            all_data[trial]['exact recognized words positions'].append(above_threshold_indices)
            all_data[trial]['exact recognized words'].append([lexicon[i] for i in above_threshold_indices])

            # Store the recongised words for the current cycle
            words_above_threshold = [x for i, x in enumerate(lexicon) if above_thresh_lexicon_np[i] == 1]
            print(f"recognized words  + {str(words_above_threshold)}")
            new_recognized_words = np.zeros(lexicon_size)

            # check whether words in stim are recognized
            # This is done by matching active words from the lexicon to slots in the stimulus.
            (matched_slots, 
             new_recognized_words, above_thresh_lexicon_np, lexicon_word_activity_np, recognized, falseguess, POSrecognition, noun_count, ver_count, 
             all_words_searched) = match_words_in_slots(trial, lexicon_size, new_recognized_words, recognized, n_words_in_stim, 
                                order_match_check, stimulus_matched_slots, stimulus, above_thresh_lexicon_np, 
                                lexicon, affixes, lexicon_word_activity_np, pm, target, pm.task_to_run, lexicon_indices_dict, all_words_searched)
            
            # Check the recognition status of the current trial based on current cycle, model parameters, 
                # and whether the target word was recognized or a false guess was made.
                # This function returns the current cycle for reaction time and a flag indicating if the trial should end.
            cycle_for_RT, end_trial = check_recognition_status(cur_cycle, pm, recognized, falseguess, lexicon_total_input_np, cycle_for_RT)
            # If the target word was recognized or a false guess was made and the trial settings dictate ending the trial, exit the loop.
            if end_trial:
                break

            # Increment the current cycle count for the next iteration.
            cur_cycle += 1

        # If the target word was recognized, determine the correctness of the recognition.
        correct, unrecognized = determine_correctness(pm, grammatical, stim, trial, identif, recognized, target)
        all_data[trial]['correct'].append(1 if correct else 0)
        if unrecognized:
            unrecognized_words.extend(unrecognized)

        # Calculate reaction time, word threshold, and word frequency for the current trial using the compute_trial_metrics function.
        reaction_time, word_threshold, word_frequency = compute_trial_metrics(cycle_for_RT, pm, target, word_thresh_dict, word_frequencies)
        # Append the calculated reaction time to the 'reaction time' list for the current trial.
        all_data[trial]['reaction time'].append(reaction_time)
        # Store the threshold for the target word in the 'word threshold' key for the current trial.
        all_data[trial]['word threshold'] = word_threshold
        # Store the frequency of the target word in the 'word frequency' key for the current trial.
        all_data[trial]['word frequency'] = word_frequency

        print("end of trial")
        print("----------------")
        print("\n")

    return all_data

def simulate_experiment_specific_tasks(pm):
    """
    Simulate an experiment based on the provided parameters (pm).

    The function follows these steps:
    1. Preprocesses stimuli to filter and organize words, removing duplicates.
    2. Processes tokens based on priming tasks and nonwords criteria.
    3. Modifies word frequencies if an affix system is specified.
    4. Computes word recognition thresholds.
    5. Maps each word from stimuli to its index in the lexicon.
    6. Computes word-to-word inhibition matrix.
    7. Displays the overlap and inhibition matrices (optional).
    8. Runs the simulation task: word recognition.

    Args:
        pm: containing all necessary parameters for the simulation.

    Returns:
        tuple: Contains all simulation data and skipped words.
    """
    print('Preparing simulation...')
    # Step 1: Process stimuli to filter and organize words, removing duplicates.
    tokens, nonwords_no_underscore, realwords, nonwords = process_and_filter_words(pm)

    # Step 2: Process tokens based on priming tasks and nonwords criteria.
    tokens, word_frequencies = process_tokens(pm, tokens, nonwords_no_underscore)

    # Step 3: Modify word frequencies based on the affix system.
    if pm.affix_system:
        word_frequencies = affix_modelling_underscores(word_frequencies)
        word_frequencies, affixes, prefixes, suffixes = process_affixes(pm, word_frequencies)
    else:
        word_frequencies = process_affixes(pm, word_frequencies)

    # Placeholder for future: POS here for future
    pred_values = get_pred_values(pm, tokens) # Deal with this later
  
    # Step 4: Build the lexicon and compute associated values.
    lexicon, lexicon_exp, lexicon_normalized_word_inhibition, max_frequency, lexicon_size, lexicon_exp_size, lexicon_word_bigrams_set = build_lexicon(tokens, word_frequencies, pm)
    
    # Step 5: Identify the real words for the experiment.
    words_from_stimuli = identify_realwords(pm, realwords, tokens)

    # Compute word recognition thresholds.
    print('Setting word recognition thresholds...')
    word_thresh_dict = compute_word_threshold(words_from_stimuli, word_frequencies, max_frequency, pm.wordfreq_p, pm.max_threshold)

    # Generate mappings for lexicon: threshold values, indices and initial activity values.
    (individual_to_lexicon_indices, 
     lexicon_thresholds_np, lexicon_index_dict, 
     lexicon_word_activity) = generate_lexicon_mappings(tokens, lexicon, word_frequencies, 
                                                        max_frequency, pm.wordfreq_p, pm.max_threshold)

    # Map each word from stimuli to its index in the lexicon array.
    lexicon_indices = map_stimuli_words_to_lexicon_indices(words_from_stimuli, lexicon) # indices as dictionary to access complex pairs later
    lexicon_indices_dict = {word: lexicon_indices[i] for i, word in enumerate(words_from_stimuli)}
    
    # Step 6: Find ngrams from the lexicon.
    print('Finding ngrams from lexicon...')
    if pm.affix_system:
        lexicon_word_bigrams, bigram_monogram_total = lexicon_bigrams(lexicon)

    # Compute the word-to-word inhibition matrix. (Including complex stem pairs)
    
    print('Computing word-to-word inhibition matrix...')
    (word_overlap_matrix, 
     word_inhibition_matrix, 
     overlap_more_than_zero, 
     overlap_more_than_one) = compute_overlap_and_inhibition_matrices(lexicon, lexicon_size, lexicon_word_bigrams, pm, 
                                                                              affixes, prefixes, suffixes, individual_to_lexicon_indices,
                                                                                lexicon_word_bigrams_set)


    # Step 7: Display the overlap and inhibition matrices (optional).
    display_overlap_and_inhibition_matrices(word_inhibition_matrix, word_overlap_matrix, pm)
    
    print("")
    print("BEGIN SIMULATION")
    print("")

    # Step 8: Run the specified simulation
    skipped_words, all_data = [],[]
    if pm.task_to_run == 'continuous_reading':
        all_data, skipped_words = reading(pm,
                                        tokens,
                                        word_inhibition_matrix,
                                        lexicon_word_ngrams,
                                        lexicon_word_index,
                                        word_thresh_dict,
                                        lexicon,
                                        pred_values,
                                        tokens_to_lexicon_indices,
                                        word_frequencies)

    else:
        all_data = word_recognition_specific_tasks(pm,
                                    word_overlap_matrix,
                                    tokens,
                                    affixes,
                                    lexicon_size,
                                    lexicon_normalized_word_inhibition,
                                    lexicon_thresholds_np,
                                    word_inhibition_matrix,
                                    lexicon_word_bigrams,
                                    bigram_monogram_total,
                                    lexicon_indices_dict,
                                    word_thresh_dict,
                                    lexicon,
                                    pred_values,
                                    word_frequencies)
    
    return all_data, skipped_words
