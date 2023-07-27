import logging
import numpy as np
import pickle
from collections import defaultdict
import math
import random
from utils import get_word_freq, get_pred_dict, set_up_inhibition_matrix, pre_process_string
from reading_components import compute_stimulus, compute_eye_position, compute_words_input, update_word_activity, \
    match_active_words_to_input_slots, compute_next_attention_position, compute_next_eye_position, \
    activate_predicted_upcoming_word
from reading_helper_functions import get_threshold, string_to_open_ngrams, \
    define_slot_matching_order, sample_from_norm_distribution, find_word_edges, update_lexicon_threshold,\
    get_blankscreen_stimulus, check_predictability

logger = logging.getLogger(__name__)

def reading(pm,tokens,text_id,word_overlap_matrix,lexicon_word_ngrams,lexicon_word_index,lexicon_thresholds,lexicon,pred_dict,freq_values):

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
    # positions in text whose thresholds/pre-activation have already been updated
    # avoid updating it every time position is in stimulus
    updated_positions = []
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
                    #'saccade_type_by_error': False}  # if the saccade type was defined due to saccade error
                    # if eye position is to be in a position other thanthat of the word middle,
                    # offset will be negative/positive (left/right)
                    # and will represent the number of letters to the new position.
                    # Its value is reset before a new saccade is performed.
                    #'offset from word center': 0}
    saccade_symbols = {'forward': ">->->->->->->->->->->->-",
                       'wordskip': ">>>>>>>>>>>>>>>>>>>>>>>>",
                       'refixation': '------------------------',
                       'regression': '<-<-<-<-<-<-<-<-<-<-<-<-'}
    tokens_to_lexicon_indices = np.zeros((total_n_words), dtype=int)
    for i, word in enumerate(tokens):
        tokens_to_lexicon_indices[i] = lexicon.index(word)

    # ---------------------- Start looping through fixations ---------------------
    while not end_of_text:

        print(f'---Fixation {fixation_counter} at position {fixation}---')

        fixation_data = defaultdict(list)

        # make sure that fixation does not go over the end of the text. Needed for continuous reading
        fixation = min(fixation, len(tokens) - 1)

        # add initial info to fixation dict
        fixation_data['foveal_word'] = tokens[fixation]
        fixation_data['foveal_word_index'] = fixation
        fixation_data['attentional_width'] = attend_width
        fixation_data['foveal_word_frequency'] = freq_values[tokens[fixation]] if tokens[fixation] in freq_values.keys() else 0
        fixation_data['foveal_word_predictability'] = pred_dict[str(fixation)][tokens[fixation]] if str(fixation) in pred_dict.keys() and tokens[fixation] in pred_dict[str(fixation)].keys() else 0
        fixation_data['foveal_word_length'] = len(tokens[fixation])
        fixation_data['foveal_word_threshold'] = lexicon_thresholds[tokens_to_lexicon_indices[fixation]]

        # saccade planning from previous fixation (or initialized values for first fixation)
        fixation_data['saccade_type'] = saccade_info['saccade_type']
        fixation_data['saccade_error'] = saccade_info['saccade_error']
        fixation_data['saccade_distance'] = saccade_info['saccade_distance']
        fixation_data['saccade_cause'] = saccade_info['saccade_cause']
        fixation_data['recognition_cycle'] = None

        # ---------------------- Define the stimulus and eye position ---------------------
        stimulus, stimulus_position, fixated_position_in_stimulus = compute_stimulus(fixation, tokens)
        eye_position = compute_eye_position(stimulus, fixated_position_in_stimulus, eye_position)
        fixation_data['stimulus'] = stimulus
        fixation_data['eye_position'] = eye_position
        print(f"Stimulus: {stimulus}\nEye position: {eye_position}")

        # ---------------------- Update attention width ---------------------
        # update attention width according to whether there was a regression in the last fixation,
        # i.e. this fixation location is a result of regression
        if fixation_data['saccade_type'] == 'regression':
            # set regression flag to know that a regression has been realized towards this position
            regression_flag[fixation] = True
            # narrow attention width by 2 letters in the case of regressions
            attend_width = max(attend_width - 2.0, pm.min_attend_width)
        else:
            # widen attention by 0.5 letters in forward saccades
            attend_width = min(attend_width + 0.5, pm.max_attend_width)

        # ---------------------- Define order of slot-matching ---------------------
        # define order to match activated words to slots in the stimulus
        # NV: the order list should reset when stimulus changes or with the first stimulus
        order_match_check = define_slot_matching_order(len(stimulus.split()), fixated_position_in_stimulus,
                                                       attend_width)
        #print(f'Order for slot-matching: {order_match_check}')

        # ---------------------- Start processing of stimulus ---------------------
        #print('Entering cycle loops to define word activity...')
        print("fix on: " + tokens[fixation] + '  attent. width: ' + str(attend_width) + ' fixwrd thresh.' + str(round(lexicon_thresholds[tokens_to_lexicon_indices[fixation]],3)))
        shift = False
        n_cycles = 0
        n_cycles_since_attent_shift = 0
        attention_position = eye_position
        # define index of letters at the words edges.
        word_edges = find_word_edges(stimulus)

        # ---------------------- Define word excitatory input ---------------------
        # compute word input using ngram excitation and inhibition (out the cycle loop because this is constant)
        n_ngrams, total_ngram_activity, all_ngrams, word_input = compute_words_input(stimulus, lexicon_word_ngrams, eye_position, attention_position, attend_width, pm)
        fixation_data['n_ngrams'] = n_ngrams
        fixation_data['total_ngram_activity'] = total_ngram_activity
        print("  input to fixwrd at first cycle: " + str(round(word_input[tokens_to_lexicon_indices[fixation]], 3)))

        # Counter n_cycles_since_attent_shift is 0 until attention shift (saccade program initiation),
        # then starts counting to 5 (because a saccade program takes 5 cycles, or 125ms.)
        while n_cycles_since_attent_shift < 5:

            # ---------------------- Update word activity per cycle ---------------------
            # Update word act with word inhibition (input remains same, so does not have to be updated)
            lexicon_word_activity, lexicon_word_inhibition = update_word_activity(lexicon_word_activity, word_overlap_matrix, pm, word_input, all_ngrams, len(lexicon))

            # update cycle info
            foveal_word_index = lexicon_word_index[tokens[fixation]]
            foveal_word_activity = lexicon_word_activity[foveal_word_index]
            fixation_data['foveal_word_activity_per_cycle'].append(foveal_word_activity)
            fixation_data['foveal_word_inhibition_per_cycle'].append(abs(lexicon_word_inhibition[foveal_word_index]))
            stim_activity = sum([lexicon_word_activity[lexicon_word_index[word]] for word in stimulus.split() if word in lexicon_word_index.keys()])
            fixation_data['stimulus_activity_per_cycle'].append(stim_activity)
            total_activity = sum(lexicon_word_activity)
            fixation_data['lexicon_activity_per_cycle'].append(total_activity)

            print('CYCLE ', str(n_cycles), '   activ @fix ', str(round(foveal_word_activity,3)), ' inhib  #@fix', str(round(lexicon_word_inhibition[foveal_word_index],6)))

            # ---------------------- Match words in lexicon to slots in input ---------------------
            # word recognition, by checking matching active wrds to slots
            recognized_word_at_position, lexicon_word_activity = \
                match_active_words_to_input_slots(order_match_check,
                                                  stimulus,
                                                  recognized_word_at_position,
                                                  lexicon_thresholds,
                                                  lexicon_word_activity,
                                                  lexicon,
                                                  pm.min_activity,
                                                  stimulus_position,
                                                  pm.word_length_similarity_constant)

            # # update threshold of n+1 or n+2 with pred value
            # if pm.prediction_flag and fixation < total_n_words-1:
            #     updated_thresh_positions, lexicon_thresholds = update_lexicon_threshold(recognized_word_at_position,
            #                                                                             fixation,
            #                                                                             tokens,
            #                                                                             updated_positions,
            #                                                                             lexicon_thresholds,
            #                                                                             pm.wordpred_p,
            #                                                                             pred_values,
            #                                                                             tokens_to_lexicon_indices)

            # after recognition, prediction-based activation of recognized word + 1
            if recognized_word_at_position.any() and fixation < total_n_words-1:
                # check whether we should pre-activate and in relation to which position (n+1 or n+2)
                position = check_predictability(recognized_word_at_position, fixation, tokens, updated_positions)
                if position and pm.prediction_flag:
                    # avoid error because of missing word in provo cloze data
                    if not (pm.prediction_flag == 'cloze' and 'provo' in pm.stim_name.lower() and position == 50 and text_id == 17):
                        lexicon_word_activity = activate_predicted_upcoming_word(position,
                                                                                 tokens[position],
                                                                                  lexicon_word_activity,
                                                                                  lexicon,
                                                                                  pred_dict,
                                                                                  pm.pred_weight)
                    updated_positions.append(position)

            # ---------------------- Make saccade decisions ---------------------
            # word selection and attention shift
            if not shift:
                # MM: on every cycle, take sample (called shift_start) out of normal distrib.
                # If cycle since fixstart > sample, make attentshift. This produces approx ex-gauss SRT
                if recognized_word_at_position[fixation]:
                    # MM: if word recog, then faster switch (norm. distrib. with <mu) than if not recog.
                    shift_start = sample_from_norm_distribution(pm.mu, pm.sigma, pm.recog_speeding,recognized=True)
                else:
                    shift_start = sample_from_norm_distribution(pm.mu, pm.sigma, pm.recog_speeding,recognized=False)
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
                                                                        pm)
                    fixation_data['foveal_word_activity_at_shift'] = fixation_data['foveal_word_activity_per_cycle'][-1]
                    print('attentpos ', attention_position)
                    # AL: attention position is None if at the end of the text and saccade is not refixation nor regression, so do not compute new words input
                    if attention_position:
                        # AL: recompute word input, using ngram excitation and inhibition, because attentshift changes bigram input
                        n_ngrams, total_ngram_activity, all_ngrams, word_input = compute_words_input(stimulus,
                                                                                                    lexicon_word_ngrams,
                                                                                                    eye_position,
                                                                                                    attention_position,
                                                                                                    attend_width, pm)
                        attention_position = np.round(attention_position)
                    print("  input after attentshift: " + str(round(word_input[tokens_to_lexicon_indices[fixation]], 3)))

            if shift:
                n_cycles_since_attent_shift += 1 # ...count cycles since attention shift

            if recognized_word_at_position[fixation] and recognized_word_at_cycle[fixation] == -1:
                # MM: here the time to recognize the word gets stored
                recognized_word_at_cycle[fixation] = n_cycles
                fixation_data['recognition_cycle'] = recognized_word_at_cycle[fixation]

            n_cycles += 1

        # out of cycle loop. After last cycle, compute fixation duration and add final values for fixated word before shift is made
        fixation_duration = n_cycles * pm.cycle_size
        fixation_data['fixation_duration'] = fixation_duration
        fixation_data['recognized_words'] = recognized_word_at_position

        print("Fixation duration: ", fixation_data['fixation duration'], " ms.")

        if recognized_word_at_position[fixation]:
            fixation_data['recognized_word_at_foveal_position'] = recognized_word_at_position[fixation]
            if recognized_word_at_position[fixation] == tokens[fixation]:
                print("Correct word recognized at fixation!")
            else:
                print(f"Wrong word recognized at fixation! (Recognized: {recognized_word_at_position[fixation]})")
        else:
            fixation_data['recognized_word_at_foveal_position'] = ""
            print("No word was recognized at fixation position")

        # add fixation dict to list of dicts
        all_data[fixation_counter] = fixation_data
        # print(fixation_data)

        fixation_counter += 1

        # compute next eye position and thus next fixation
        if attention_position:
            fixation, eye_position, saccade_info = compute_next_eye_position(pm, attention_position, eye_position, fixation, fixated_position_in_stimulus, word_edges, saccade_info)

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
        elif not attention_position:
            end_of_text = True
            continue
        else:
            print(saccade_symbols[saccade_info['saccade_type']])

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
                                                                                  all_ngrams,
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

    print('Preparing simulation...')
    tokens = [token for stimulus in pm.stim_all for token in stimulus.split(' ') if token != '']

    if pm.is_priming_task:
        tokens.extend([token for stimulus in list(pm.stim["prime"]) for token in stimulus.split(' ')])

    tokens = [pre_process_string(token) for token in tokens]
    # remove empty strings which were once punctuations
    tokens = [token for token in tokens if token != '']
    word_frequencies = get_word_freq(pm, set(tokens))
    max_frequency = max(word_frequencies.values())
    lexicon = list(set(tokens) | set(word_frequencies.keys()))
    lexicon = [pre_process_string(word) for word in lexicon]

    # write out lexicon for consulting purposes
    lexicon_filename = '../data/processed/lexicon.pkl'
    with open(lexicon_filename, "wb") as f:
        pickle.dump(lexicon, f)

    print('Setting word recognition thresholds...')
    # define word recognition thresholds
    word_thresh_dict = {}
    for word in lexicon:
        if pm.frequency_flag:
            word_thresh_dict[word] = get_threshold(word,
                                                   word_frequencies,
                                                   max_frequency,
                                                   pm.wordfreq_p,
                                                   pm.max_threshold)

    # lexicon indices for each word of text
    total_n_words = len(tokens)
    tokens_to_lexicon_indices = np.zeros((total_n_words), dtype=int)
    for i, word in enumerate(tokens):
        tokens_to_lexicon_indices[i] = lexicon.index(word)

    print('Finding ngrams from lexicon...')
    # lexicon bigram dict
    lexicon_word_ngrams = {}
    lexicon_word_index = {}
    for i, word in enumerate(lexicon):
        # AL: weights and locations are not used for lexicon, only the ngrams of the words in the lexicon for comparing them later with the ngrams activated in stimulus.
        all_word_ngrams, weights, locations = string_to_open_ngrams(word,pm.bigram_gap)
        lexicon_word_ngrams[word] = all_word_ngrams
        lexicon_word_index[word] = i

    print('Computing word-to-word inhibition matrix...')
    # set up word-to-word inhibition matrix
    word_inhibition_matrix = set_up_inhibition_matrix(pm, lexicon, lexicon_word_ngrams)
    print("Inhibition grid ready.")

    # recognition threshold for each word in lexicon
    lexicon_thresholds = np.zeros((len(lexicon)), dtype=float)
    # initialize thresholds with values based frequency, if dict has been filled (if frequency flag)
    if word_thresh_dict != {}:
        for i, word in enumerate(lexicon):
            lexicon_thresholds[i] = word_thresh_dict[word]

    print("")
    print("BEGIN SIMULATION")
    print("")

    # read text/trials
    all_data = defaultdict()

    for sim_number in range(pm.number_of_simulations):

        if pm.task_to_run == 'continuous_reading':

            texts_simulations = defaultdict()

            # AL: if language model, generate new predictions with a new seed for every x simulations
            word_predictions = get_pred_dict(pm, lexicon)

            for i, text in enumerate(pm.stim_all):

                text_tokens = [pre_process_string(token) for token in text.split()]
                text_data = reading(pm,
                                    text_tokens,
                                    i,
                                    word_inhibition_matrix,
                                    lexicon_word_ngrams,
                                    lexicon_word_index,
                                    lexicon_thresholds,
                                    lexicon,
                                    word_predictions[str(i)],
                                    word_frequencies)
                texts_simulations[i] = text_data

            all_data[sim_number] = texts_simulations

        else:
            all_data = word_recognition(pm,
                                        word_inhibition_matrix,
                                        lexicon_word_ngrams,
                                        lexicon_word_index,
                                        word_thresh_dict,
                                        lexicon,
                                        word_frequencies)

        print(f'END of SIMULATION {sim_number}\n')

    print(f'THE END')

    return all_data
