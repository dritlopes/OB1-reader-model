import logging
import numpy as np
import pickle
from collections import defaultdict
import math
from utils import get_word_freq, get_pred_values, check_previous_inhibition_matrix, pre_process_string
from reading_processes import compute_stimulus, compute_eye_position, compute_words_input, update_word_activity, \
    match_active_words_to_input_slots, compute_next_attention_position, compute_next_eye_position
from reading_functions import get_threshold, string_to_open_ngrams, build_word_inhibition_matrix,\
    define_slot_matching_order, sample_from_norm_distribution, find_word_edges, update_lexicon_threshold,\
    get_blankscreen_stimulus

logger = logging.getLogger(__name__)

def reading(pm,tokens,word_overlap_matrix,lexicon_word_ngrams,lexicon_word_index,lexicon_thresholds_dict,lexicon,pred_values,tokens_to_lexicon_indices,freq_values):

    # information computed for each fixation
    all_data = {}
    # set to true when end of text is reached
    end_of_text = False
    # the element of fixation in the text. It goes backwards in case of regression
    fixation = 0
    # +1 with every next fixation
    fixation_counter = 0
    # initialise attention window size
    attend_width = pm.attend_width
    # total number of tokens in input
    total_n_words = len(tokens)
    # word activity for word in lexicon
    lexicon_word_activity = np.zeros((len(lexicon)), dtype=float)
    # recognition threshold for each word in lexicon
    lexicon_thresholds = np.zeros((len(lexicon)), dtype=float)
    # positions in text whose thresholds have already been updated, avoid updating it every time position is in stimulus
    updated_thresh_positions = []
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
                    'saccade_cause': 0,  # for wordskip and refixation, extra info on cause of saccade
                    'saccade_type_by_error': False,  # if the saccade type was defined due to saccade error
                    # if eye position is to be in a position other thanthat of the word middle,
                    # offset will be negative/positive (left/right)
                    # and will represent the number of letters to the new position.
                    # Its value is reset before a new saccade is performed.
                    'offset_from_word_center': 0}

    # initialize thresholds with values based frequency, if dict has been filled (if frequency flag)
    if lexicon_thresholds_dict != {}:
        for i, word in enumerate(lexicon):
            lexicon_thresholds[i] = lexicon_thresholds_dict[word]

    # ---------------------- Start looping through fixations ---------------------
    while not end_of_text:

        print(f'---Fixation {fixation_counter} at position/trial {fixation}---')

        fixation_data = defaultdict(list)

        # make sure that fixation does not go over the end of the text. Needed for continuous reading
        fixation = min(fixation, len(tokens) - 1)

        # add initial info to fixation dict
        fixation_data['foveal word'] = tokens[fixation]
        fixation_data['foveal word index'] = fixation
        fixation_data['attentional width'] = attend_width
        fixation_data['foveal word frequency'] = freq_values[tokens[fixation]] if tokens[fixation] in freq_values.keys() else 0
        fixation_data['foveal word predictability'] = pred_values[str(fixation)] if str(fixation) in pred_values.keys() else 0
        fixation_data['foveal word threshold'] = lexicon_thresholds[tokens_to_lexicon_indices[fixation]]
        fixation_data['offset'] = saccade_info['offset_from_word_center']
        fixation_data['saccade_type'] = saccade_info['saccade_type']
        fixation_data['saccade error'] = saccade_info['saccade_error']
        fixation_data['saccade distance'] = saccade_info['saccade_distance']
        fixation_data['saccade_cause'] = saccade_info['saccade_cause']
        fixation_data['saccade_type_by_error'] = saccade_info['saccade_type_by_error']

        # ---------------------- Define the stimulus and eye position ---------------------
        stimulus, stimulus_position, fixated_position_in_stimulus = compute_stimulus(fixation, tokens)
        eye_position = compute_eye_position(stimulus, fixated_position_in_stimulus, saccade_info['offset_from_word_center'])
        fixation_data['stimulus'] = stimulus
        fixation_data['eye position'] = eye_position
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
        # print("  input to fixwrd at first cycle: " + str(round(word_input[tokens_to_lexicon_indices[fixation]], 3)))

        # Counter n_cycles_since_attent_shift is 0 until attention shift (saccade program initiation),
        # then starts counting to 5 (because a saccade program takes 5 cycles, or 125ms.)
        while n_cycles_since_attent_shift < 5:

            # ---------------------- Update word activity per cycle ---------------------
            # Update word act with word inhibition (input remains same, so does not have to be updated)
            lexicon_word_activity, lexicon_word_inhibition = update_word_activity(lexicon_word_activity, word_overlap_matrix, pm, word_input, all_ngrams, len(lexicon))

            # update cycle info
            act_of_ist = lexicon_word_activity[lexicon_word_index['ist']]
            foveal_word_index = lexicon_word_index[tokens[fixation]]
            foveal_word_activity = lexicon_word_activity[foveal_word_index]
            print('CYCLE ', str(n_cycles), '   activ @fix ', str(round(foveal_word_activity,3)), '   inhib of ist', str(round(lexicon_word_inhibition[lexicon_word_index['ist']],3))) #@fix', str(round(lexicon_word_inhibition[foveal_word_index],3)))
            #print('        and act. of Die', str(round(lexicon_word_activity[lexicon_word_index[tokens[0]]],3)))
            fixation_data['foveal word activity per cycle'].append(foveal_word_activity)
            fixation_data['foveal word-to-word inhibition per cycle'].append(abs(lexicon_word_inhibition[foveal_word_index]))
            stim_activity = sum([lexicon_word_activity[lexicon_word_index[word]] for word in stimulus.split() if word in lexicon_word_index.keys()])
            fixation_data['stimulus activity per cycle'].append(stim_activity)
            total_activity = sum(lexicon_word_activity)
            fixation_data['lexicon activity per cycle'].append(total_activity)

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

            # update threshold of n+1 or n+2 with pred value
            if pm.prediction_flag and fixation < total_n_words-1:
                updated_thresh_positions, lexicon_thresholds = update_lexicon_threshold(recognized_word_at_position,
                                                                                        fixation,
                                                                                        tokens,
                                                                                        updated_thresh_positions,
                                                                                        lexicon_thresholds,
                                                                                        pm.wordpred_p,
                                                                                        pred_values,
                                                                                        tokens_to_lexicon_indices)
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
                                    'saccade_cause': 0,
                                    'saccade_type_by_error': False,
                                    'offset_from_word_center': 0}
                    attention_position, saccade_info = compute_next_attention_position(all_data,
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
                                                                                        saccade_info)
                    fixation_data['foveal word activity at shift'] = fixation_data['foveal word activity per cycle'][-1]
                    #print('attentpos ', attention_position)
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
                fixation_data['recognition cycle'] = recognized_word_at_cycle[fixation]

            n_cycles += 1

        # out of cycle loop. After last cycle, compute fixation duration and add final values for fixated word before shift is made
        fixation_duration = n_cycles * pm.cycle_size
        fixation_data['fixation duration'] = fixation_duration
        fixation_data['recognized word at position'] = recognized_word_at_position

        # add fixation dict to list of dicts
        all_data[fixation_counter] = fixation_data

        print("Fixation duration: ", fixation_data['fixation duration'], " ms.")
        if recognized_word_at_position[fixation]:
            if recognized_word_at_position[fixation] == tokens[fixation]:
                print("Correct word recognized at fixation!")
            else:
                print(f"Wrong word recognized at fixation! (Recognized: {recognized_word_at_position[fixation]})")
        else:
            print("No word was recognized at fixation position")

        fixation_counter += 1

        # Check if end of text is reached AL: if fixation on last word and next saccade not refixation nor regression
        if fixation == total_n_words - 1 and saccade_info['saccade_type'] not in ['refixation', 'regression']:
            end_of_text = True
            print(recognized_word_at_position)
            print("END REACHED!")
            continue

        #if fixation_counter > 6: exit()
        # if end of text is not yet reached, compute next eye position and thus next fixation
        fixation, next_eye_position, saccade_info = compute_next_eye_position(pm, saccade_info, eye_position, stimulus, fixation, total_n_words, word_edges, fixated_position_in_stimulus)

    # # register words in text in which no word in lexicon reaches recognition threshold
    # unrecognized_words = dict()
    # for position in range(total_n_words):
    #     if not recognized_word_at_position[position]:
    #         unrecognized_words[position] = tokens[position]

    return all_data

def word_recognition(pm,tokens,word_inhibition_matrix,lexicon_word_ngrams,lexicon_word_index,word_thresh_dict,lexicon,pred_values,tokens_to_lexicon_indices,word_frequencies):

    # information computed for each fixation
    all_data = {}
    # data frame with stimulus info
    stim = pm.stim
    # initialise attention window size
    attend_width = pm.attend_width
    # total number of tokens in input
    total_n_words = len(tokens)
    # word activity for each word in lexicon
    lexicon_word_activity = np.zeros((len(lexicon)), dtype=float)
    # recognition threshold for each word in lexicon
    lexicon_thresholds = np.zeros((len(lexicon)), dtype=float)
    # positions in text whose thresholds have already been updated
    updated_thresh_positions = []

    for trial in range(0, len(stim['all'])):

        # all_data[trial] = {'stimulus': [],
        #                    'prime': [],  # NV: added prime
        #                    'target': [],
        #                    'condition': [],
        #                    'cycle': [],
        #                    'lexicon activity per cycle': [],
        #                    'stimulus activity per cycle': [],
        #                    'target activity per cycle': [],
        #                    'bigram activity per cycle': [],
        #                    'ngrams': [],
        #                    # 'recognized words indices': [],
        #                    # 'attentional width': attendWidth,
        #                    'exact recognized words positions': [],
        #                    'exact recognized words': [],
        #                    'eye position': EyePosition,
        #                    'attention position': AttentionPosition,
        #                    'word threshold': 0,
        #                    'word frequency': 0,
        #                    'word predictability': 0,
        #                    'reaction time': [],
        #                    'correct': [],
        #                    'POS': [],
        #                    'position': [],
        #                    'inhibition_value': pm.word_inhibition,  # NV: info for plots in notebook
        #                    'wordlen_threshold': pm.word_length_similarity_constant,
        #                    'target_inhib': [],
        #                    'error_rate': 0}  # NV: info for plots in notebook

        # ---------------------- Define the trial stimuli ---------------------
        stimulus = stim['all'][trial]
        fixated_position_in_stim = math.floor(len(stimulus.split(' '))/2)
        eye_position = np.round(len(stimulus) // 2)
        target = stim['target'][trial]
        condition = stim['condition'][trial]
        if pm.is_priming_task:
            prime = stim['prime'][trial]

        # ---------------------- Start processing stimuli ---------------------
        n_cycles = 0
        attention_position = eye_position
        stimuli = []
        recognized = False
        # init activity matrix with min activity. Assumption that each trial is independent.
        lexicon_word_activity[lexicon_word_activity < pm.min_activity] = pm.min_activity

        # keep processing stimuli as long as trial lasts
        while n_cycles < pm.totalcycles:

            # stimulus may change within a trial to blankscreen, prime
            if n_cycles < pm.blankscreen_cycles_begin or n_cycles > pm.totalcycles - pm.blankscreen_cycles_end:
                stimulus = get_blankscreen_stimulus(pm.blankscreen_type)
            elif pm.is_priming_task and n_cycles < (pm.blankscreen_cycles_begin+pm.ncyclesprime):
                stimulus = prime
            else:
                stimulus = stim['all'][trial]
            stimuli.append(stimulus)

            # define order for slot matching. Computed within cycle loop bcs stimulus may change within trial
            order_match_check = define_slot_matching_order(stimulus,fixated_position_in_stim,attend_width)

            # compute word excitatory input given stimulus
            n_ngrams, total_ngram_activity, all_ngrams, word_input = compute_words_input(stimulus,
                                                                                         lexicon_word_ngrams,
                                                                                         eye_position,
                                                                                         attention_position,
                                                                                         attend_width,
                                                                                         pm)
            # update word activity using word-to-word inhibition and decay
            lexicon_word_activity, lexicon_word_inhibition = update_word_activity(lexicon_word_activity,
                                                                                  word_inhibition_matrix,
                                                                                  pm,
                                                                                  word_input,
                                                                                  all_ngrams,
                                                                                  len(lexicon))
            # word recognition, by checking matching active wrds to slots
            recognized, lexicon_word_activity = \
                match_active_words_to_input_slots(order_match_check,
                                                  stimulus,
                                                  recognized,
                                                  lexicon_thresholds,
                                                  lexicon_word_activity,
                                                  lexicon,
                                                  pm.min_activity,
                                                  None,
                                                  pm.word_length_similarity_constant)

            # register cycle of recognition
            if recognized:
                recog_cycle = n_cycles

            n_cycles += 1

        # compute reaction time
        reaction_time = ((recog_cycle + 1 - pm.blankscreen_cycles_begin) * pm.cycle_size) + 300
        print("reaction time: " + str(reaction_time) + " ms")
        print("end of trial")
        print("----------------")
        print("\n")

    return all_data

def simulate_experiment(pm):

    print('Preparing simulation...')

    if type(pm.stim_all) == str:
        tokens = pm.stim_all.split(' ')
    else:
        tokens = [token for stimulus in pm.stim_all for token in stimulus.split(' ')]

    if pm.is_priming_task:
        tokens.extend([token for stimulus in list(pm.stim["prime"]) for token in stimulus.split(' ')])

    tokens = [pre_process_string(token) for token in tokens]
    word_frequencies = get_word_freq(pm, set([token.lower() for token in tokens]))
    pred_values = get_pred_values(pm, set(tokens))
    max_frequency = max(word_frequencies.values())
    lexicon = list(set(tokens) | set(word_frequencies.keys()))

    # write out lexicon for consulting purposes
    lexicon_file_name = '../data/Lexicon.dat'
    with open(lexicon_file_name, "wb") as f:
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
    previous_matrix_usable = check_previous_inhibition_matrix(pm,lexicon,lexicon_word_ngrams)
    if previous_matrix_usable:
        with open('../data/Inhibition_matrix_previous.dat', "rb") as f:
            word_inhibition_matrix = pickle.load(f)
    else:
        word_inhibition_matrix = build_word_inhibition_matrix(lexicon,lexicon_word_ngrams,pm,tokens_to_lexicon_indices)
    print("Inhibition grid ready.")

    print("")
    print("BEGIN SIMULATION")
    print("")

    # read text/trials
    if pm.task_to_run == 'continuous reading':
        all_data = reading(pm,
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
        all_data = word_recognition(pm,
                                    tokens,
                                    word_inhibition_matrix,
                                    lexicon_word_ngrams,
                                    lexicon_word_index,
                                    word_thresh_dict,
                                    lexicon,
                                    pred_values,
                                    tokens_to_lexicon_indices,
                                    word_frequencies)

    return all_data
