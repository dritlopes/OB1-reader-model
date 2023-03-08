import logging
import numpy as np
import pickle
from collections import defaultdict
from utils import get_word_freq, get_pred_values, check_previous_inhibition_matrix, pre_process_string
from reading_functions import get_threshold, string_to_open_ngrams, build_word_inhibition_matrix, cal_ngram_exc_input, define_slot_matching_order, is_similar_word_length, \
    sample_from_norm_distribution, find_word_edges, get_midword_position_for_surrounding_word, calc_word_attention_right, update_lexicon_threshold, calc_saccade_error,\
    check_previous_refixations_at_position

logger = logging.getLogger(__name__)

def compute_stimulus(fixation, tokens):

    """
    Given fixation position in text and the text tokens, find the stimulus for a given fixation.
    The stimulus is normally made of 5 words: n-2 to n+2 (n being the fixated word).
    :return: the stimulus, the position of each word in the stimulus in relation to the text,
    and the position of the fixated word in relation to the stimulus.
    """

    # assuming stimulus default is n-2 to n+2
    start_window = fixation - 2
    end_window = fixation + 2
    # only add position if after text begin and below text length
    stimulus_position = [i for i in range(start_window,end_window+1) if i >= 0 and i < len(tokens)]
    stimulus = ' '.join([tokens[i] for i in stimulus_position])
    fixated_position_stimulus = stimulus_position.index(fixation)

    return stimulus, stimulus_position, fixated_position_stimulus

def compute_eye_position(stimulus, fixated_position_stimulus, offset_from_word_center):

    """
    Given the stimulus during a fixation, find where the eye is positioned in relation to the stimulus.
    :return: the index of the character the eyes are fixating at in the stimulus (in number of characters).
    """

    stimulus = stimulus.split(' ')
    center_of_fixation = round(len(stimulus[fixated_position_stimulus]) * 0.5)
    # find length of stimulus (in characters) up until fixated word
    len_till_fix = sum([len(token)+1 for token in stimulus[:fixated_position_stimulus]])
    eye_position = len_till_fix + center_of_fixation + offset_from_word_center

    return int(np.round(eye_position))

def compute_ngram_activity(stimulus, eye_position, attention_position, attend_width, letPerDeg, attention_skew, gap):

    """
    Initialize word activity based on ngram excitatory input.
    :return: dict with ngram as keys and excitatory input as value.
    """

    unit_activations = {}
    all_ngrams,all_weights,all_locations = string_to_open_ngrams(stimulus,gap)

    for ngram, weight, location in zip(all_ngrams,all_weights,all_locations):
        activation = cal_ngram_exc_input(location, weight, eye_position, attention_position,
                                         attend_width, letPerDeg, attention_skew)
        # AL: a ngram that appears more than once in the simulus get summed activation
        if ngram in unit_activations.keys():
            unit_activations[ngram] = unit_activations[ngram] + activation
        else:
            unit_activations[ngram] = activation

    return unit_activations

def compute_words_input(stimulus, lexicon_word_ngrams, eye_position, attention_position, attend_width, pm):

    """
    Calculate activity for each word in the lexicon given the excitatory input from all ngrams in the stimulus.
    :return: word_input (array) with the resulting activity for each word in the lexicon,
    all_ngrams (list) with the number of ngrams per word in the lexicon,
    total_ngram_activity (int) as the total activity ngrams in the input resonate in the lexicon,
    n_ngrams (int) as the total number of ngrams in the input.
    """

    lexicon_size = len(lexicon_word_ngrams.keys())
    word_input = np.zeros((lexicon_size), dtype=float)

    # define ngram activity given stimulus
    unit_activations = compute_ngram_activity(stimulus, eye_position,
                                              attention_position, attend_width, pm.letPerDeg,
                                              pm.attention_skew, pm.bigram_gap)
    total_ngram_activity = sum(unit_activations.values())
    n_ngrams = len(unit_activations.keys())

    # compute word input according to ngram excitation and inhibition
    # all stimulus bigrams used, therefore the same bigram inhibition for each word of lexicon
    # (excit is specific to word, inhib same for all)
    ngram_inhibition_input = sum(unit_activations.values()) * pm.bigram_to_word_inhibition
    for lexicon_ix, lexicon_word in enumerate(lexicon_word_ngrams.keys()):
        word_excitation_input = 0
        # ngram (bigram & monogram) activations
        ngram_intersect_list = set(unit_activations.keys()).intersection(set(lexicon_word_ngrams[lexicon_word]))
        for ngram in ngram_intersect_list:
            word_excitation_input += pm.bigram_to_word_excitation * unit_activations[ngram]
        word_input[lexicon_ix] = word_excitation_input + ngram_inhibition_input

    # normalize based on number of ngrams in lexicon
    # MM: Add discounted_Ngrams to nr ngrams. Decreases input to short words
    # to compensate for fact that higher prop of their bigrams have higher wgt because edges
    all_ngrams = [len(ngrams) for ngrams in lexicon_word_ngrams.values()]
    word_input = word_input / (np.array(all_ngrams) + pm.discounted_Ngrams)

    return n_ngrams, total_ngram_activity, all_ngrams, word_input

def update_word_activity(lexicon_word_activity, word_overlap_matrix, pm, word_input, all_ngrams, lexicon_size):

    """
    In each processing cycle, re-compute word activity using word-to-word inhibition and decay.
    :return: lexicon_word_activity (array) with updated activity for each word in the lexicon,
    lexicon_word_inhibition (array) with total inhibition for each word in the lexicon.
    """

    # NV: the more active a certain word is, the more inhibition it will execute on its peers
    # -> activity is multiplied by inhibition constant.
    # NV: then, this inhibition value is weighed by how much overlap there is between that word and every other.
    lexicon_normalized_word_inhibition = (100.0 / lexicon_size) * pm.word_inhibition
    # find which words are active
    lexicon_active_words = np.zeros((lexicon_size), dtype=bool)
    lexicon_active_words[(lexicon_word_activity > 0.0) | (word_input > 0.0)] = True
    overlap_select = word_overlap_matrix[:, (lexicon_active_words == True)]
    lexicon_select = (lexicon_word_activity + word_input)[
                         (lexicon_active_words == True)] * lexicon_normalized_word_inhibition
    # This concentrates inhibition on the words that have most overlap and are most active
    #lexicon_word_inhibition = np.dot((overlap_select ** 2), -(lexicon_select ** 2)) / np.array(len(set(all_ngrams)))
    lexicon_word_inhibition = np.dot((overlap_select ** 2), -(lexicon_select ** 2)) / np.array(len(set(all_ngrams))**2)
    # Combine word inhibition and input, and update word activity
    lexicon_total_input = np.add(lexicon_word_inhibition, word_input)

    # final computation of word activity
    # pm.decay has a neg value, that's why it's here added, not subtracted
    lexicon_word_activity_change = ((pm.max_activity - lexicon_word_activity) * lexicon_total_input) + \
                                   ((lexicon_word_activity - pm.min_activity) * pm.decay)
    lexicon_word_activity = np.add(lexicon_word_activity, lexicon_word_activity_change)
    # correct activity beyond minimum and maximum activity to min and max
    lexicon_word_activity[lexicon_word_activity < pm.min_activity] = pm.min_activity
    lexicon_word_activity[lexicon_word_activity > pm.max_activity] = pm.max_activity

    return lexicon_word_activity, lexicon_word_inhibition

def match_active_words_to_input_slots(order_match_check, stimulus, recognized_word_at_position, above_thresh_lexicon, lexicon_word_activity, lexicon, min_activity, stimulus_position, len_sim_const):

    """
    Match active words to spatio-topic representation. Fill in the stops in the stimulus.
    The winner is the word with the highest activity above recognition threshold and of similar length.
    :return: recognized_word_at_position is the updated array of recognized words in each text position,
    lexicon_word_activity is the updated array with activity of each word in the lexicon,
    above_thresh_lexicon is the updated array with which words have activity above threshold.
    """

    for slot_to_check in range(len(order_match_check)):
        # slot_num is the slot in the stim (spot of still-unrecogn word) that we're checking
        slot_num = order_match_check[slot_to_check]
        word_index = stimulus_position[slot_num]
        # if the slot has not yet been filled
        if not recognized_word_at_position[word_index]:
            # Check words that have the same length as word in the slot we're now looking for
            word_searched = stimulus.split()[slot_num]
            # MM: recognWrdsFittingLen_np: array with 1=wrd act above threshold, & approx same len
            # as to-be-recogn wrd (with 'len_sim_const' margin), 0=otherwise
            similar_length = np.array([int(is_similar_word_length(len(x.replace('_', '')),
                                                                  len(word_searched), len_sim_const)) for x in lexicon])
            recognized_words_fit_len = above_thresh_lexicon * similar_length
            # if at least one word matches (act above threshold and similar length)
            if sum(recognized_words_fit_len):
                # Find the word with the highest activation in all words that have a similar length
                highest = np.argmax(recognized_words_fit_len * lexicon_word_activity)
                highest_word = lexicon[highest]
                print('word in input: ', word_searched, '      one w. highest act: ', highest_word)
                # The winner is matched to the slot,
                # and its activity is reset to minimum to not have it matched to other words
                recognized_word_at_position[word_index] = highest_word
                lexicon_word_activity[highest] = min_activity

    return recognized_word_at_position, lexicon_word_activity

def compute_next_attention_position(all_data,tokens,fixation,word_edges,fixated_position_in_stimulus,regression_flag,recognized_word_at_position,lexicon_word_activity,eye_position,fixation_counter,attention_position,attend_width,fix_lexicon_index,pm,saccade_info):

    """
    Define where attention should be moved to next based on recognition of words in current stimulus and the visual
    salience of the words to the right of fixation.
    :return: the next attention position as the index of the letter in the word programmed to be fixated next,
    and the updated saccade info based on the next attention position.
    """

    # Define target of next fixation relative to fixated word n (i.e. 0=next fix on word n, -1=fix on n-1, etc). Default is 1 (= to word n+1)
    next_fixation = 1
    refix_size = pm.refix_size

    # regression: if the current fixation was a regression and next word has been recognized, move eyes to n+2 to resume reading
    if regression_flag[fixation] and recognized_word_at_position[fixation + 1]:
        next_fixation = 2

    # regression: check whether previous word was recognized or there was already a regression performed. If not: regress
    elif fixation > 1 and not recognized_word_at_position[fixation - 1] and not regression_flag[fixation - 1]:
        next_fixation = -1

    # refixation: refixate if the foveal word is not recognized but is still being processed
    elif (not recognized_word_at_position[fixation]) and (lexicon_word_activity[fix_lexicon_index] > 0):
        # # AL: only allows 3 consecutive refixations on the same word to avoid infinitely refixating if no word reaches threshold recognition at a given position
        refixate = check_previous_refixations_at_position(all_data,fixation,fixation_counter,max_n_refix=3)
        if refixate:
            word_reminder_length = word_edges[fixated_position_in_stimulus][1] - eye_position
            if word_reminder_length > 0:
                next_fixation = 0
                if fixation_counter - 1 in all_data.keys():
                    if not all_data[fixation_counter - 1]['refixated']:
                        refix_size = np.round(word_reminder_length * refix_size)

    # forward saccade: perform normal forward saccade (unless at the last position in the text)
    elif fixation < (len(tokens) - 1):
        word_attention_right = calc_word_attention_right(word_edges,
                                                         eye_position,
                                                         attention_position,
                                                         attend_width,
                                                         pm.salience_position,
                                                         pm.attention_skew,
                                                         pm.letPerDeg,
                                                         fixated_position_in_stimulus)
        next_fixation = word_attention_right.index(max(word_attention_right))

    # AL: Calculate next attention position based on next fixation estimate = 0: refixate, 1: forward, 2: wordskip, -1: regression
    if next_fixation == 0:
        # MM: if we're refixating same word because it has highest attentwgt AL: or not being recognized whilst processed
        # ...use first refixation middle of remaining half as refixation stepsize
        fixation_first_position_right_to_eye = eye_position + 1 if eye_position + 1 < len(tokens) else eye_position
        attention_position = fixation_first_position_right_to_eye + refix_size
    else:
        assert (next_fixation in [-1, 1, 2])
        attention_position = get_midword_position_for_surrounding_word(next_fixation, word_edges, fixated_position_in_stimulus)

    # AL: Update saccade info
    if next_fixation == 1: # AL: next fixation n + 1
        saccade_info['saccade_type'] = 'forward'
    elif next_fixation == 2: # AL: next fixation n + 2
        saccade_info['saccade_type'] = 'wordskip'
        if regression_flag[fixation]:
            saccade_info['saccade_cause'] = 2 # AL: bcs n resulted from regression and n + 1 has been recognized
        else:
            saccade_info['saccade_cause'] = 1 # AL: bcs n + 2 has highest attwght (letter excitation)
    elif next_fixation == -1:
        saccade_info['saccade_type'] = 'regression'
    elif next_fixation == 0:
        saccade_info['saccade_type'] = 'refixation'
        if not recognized_word_at_position[fixation]:
            saccade_info['saccade_cause'] = 1 # AL: bcs fixated word has not been recognized
        else:
            saccade_info['saccade_cause'] = 2 # AL: bcs right of fixated word has highest attwght (letter excitation)

    # AL: saccade distance is next attention position minus the current eye position
    if attention_position:
        saccade_info['saccade_distance'] = attention_position - eye_position

    return attention_position, saccade_info

def compute_next_eye_position(pm, saccade_info, eye_position, stimulus, fixation, total_n_words, word_edges, fixated_position_in_stimulus, verbose=True):

    """
    This function computes next eye position and next offset from word center using saccade distance
    (defined by next attention position and current eye position) plus a saccade error.
    Importantly, it corrects the offset to prevent too short or too long saccades.
    :return: the next fixation, the next eye position and the updated saccade info
    """

    saccade_distance = saccade_info['saccade_distance']
    offset_from_word_center = saccade_info['offset_from_word_center']

    # normal random error based on difference with optimal saccade distance
    saccade_error = calc_saccade_error(saccade_distance,
                                       pm.sacc_optimal_distance,
                                       pm.saccErr_scaler,
                                       pm.saccErr_sigma,
                                       pm.saccErr_sigma_scaler,
                                       pm.use_saccade_error)

    saccade_distance = saccade_distance + saccade_error
    offset_from_word_center = offset_from_word_center + saccade_error

    # compute the position of next fixation
    next_eye_position = int(np.round(eye_position + saccade_distance))
    if next_eye_position >= len(stimulus) - 1:
        next_eye_position = len(stimulus) - 2

    # Calculating the actual saccade type
    # Regression
    if next_eye_position < word_edges[fixated_position_in_stimulus][0]:
        # AL: if eye at space right to word n - 1, offset corrects eye to word n - 1
        if next_eye_position > word_edges[fixated_position_in_stimulus-1][1]:
            offset_from_word_center -= 1
        # AL: if eye at space left to word n - 1, offset corrects eye to word n - 1
        if next_eye_position < word_edges[fixated_position_in_stimulus-1][0]:
            center_position = get_midword_position_for_surrounding_word(-1,word_edges,fixated_position_in_stimulus)
            offset_from_word_center = center_position - word_edges[fixated_position_in_stimulus-1][0]
        fixation -= 1
        saccade_info['saccade_type'] = 'regression'

    # Forward (include space between n and n+2)
    # AL: next eye position is too long (between space after last letter of fixated word and last letter of word n + 1)
    elif ((fixation < total_n_words - 1)
          and (next_eye_position > word_edges[fixated_position_in_stimulus][1])
          and (next_eye_position <= (word_edges[fixated_position_in_stimulus+1][1]))):
        # When saccade too short due to saccade error recalculate offset for n+1 (old offset is for N or N+2)
        if saccade_info['saccade_type'] in ['wordskip','refixation']:
            center_position = get_midword_position_for_surrounding_word(1,word_edges,fixated_position_in_stimulus)
            offset_from_word_center = next_eye_position - center_position
            # AL: forward due to saccade error
            saccade_info['saccade_type_by_error'] = True
        # Eye at (n+0 <-> n+1) space position
        if next_eye_position < word_edges[fixated_position_in_stimulus+1][0]:
            offset_from_word_center += 1
        fixation += 1
        saccade_info['saccade_type'] = 'forward'

    # Wordskip
    # AL: next eye position after last letter of word n+1 and before 3 chars after last letter of word n+2
    elif ((fixation < total_n_words - 2)
          and (next_eye_position > word_edges[fixated_position_in_stimulus+1][1])
          and (next_eye_position <= word_edges[fixated_position_in_stimulus+2][1] + 2)):
        # AL: When saccade is too short, recalculate offset to correct eye position to n+2 (old offset is for n or n+1)
        if saccade_info['saccade_type'] in ['forward','refixation']:
            # recalculate offset for n+2, todo check for errors
            center_position = get_midword_position_for_surrounding_word(2,word_edges,fixated_position_in_stimulus)
            offset_from_word_center = next_eye_position - center_position
            # AL: wordskip due to saccade error
            saccade_info['saccade_type_by_error'] = True
        # Eye at (n+1 <-> n+2) space position
        if next_eye_position < word_edges[fixated_position_in_stimulus+2][0]:
            offset_from_word_center += 1
        # Eye at (> n+2) space position
        elif next_eye_position > word_edges[fixated_position_in_stimulus+2][1]:
            offset_from_word_center -= (next_eye_position - word_edges[fixated_position_in_stimulus+2][1])
        fixation += 2
        saccade_info['saccade_type'] = 'wordskip'

    # Refixation due to saccade error
    # AL: weird to me, isn't this converting all saccades which were not refixation not due by error to refixation due by error?
    elif saccade_info['saccade_type'] != 'refixation':
        # TODO find out if not regression is necessary
        center_position = np.round(word_edges[fixated_position_in_stimulus][0] +
                                  ((word_edges[fixated_position_in_stimulus][1] - word_edges[fixated_position_in_stimulus][0]) / 2.))
        offset_from_word_center = next_eye_position - center_position
        saccade_info['saccade_type_by_error'] = True
        saccade_info['saccade_type'] = 'refixation'

    # update saccade info
    saccade_info['offset_from_word_center'] = float(offset_from_word_center)
    saccade_info['saccade_distance'] = float(saccade_distance)
    saccade_info['saccade_error'] = float(saccade_error)

    if verbose:
        if saccade_info['saccade_type'] == 'forward':
            print(">->->->->->->->->->->->-")
        elif saccade_info['saccade_type'] == 'wordskip':
            print(">>>>>>>>>>>>>>>>>>>>>>>>")
        elif saccade_info['saccade_type'] == 'refixation':
            print("------------------------")
        elif saccade_info['saccade_type'] == 'regression':
            print("<-<-<-<-<-<-<-<-<-<-<-<-")

    return fixation, next_eye_position, saccade_info

def reading(pm,tokens,word_overlap_matrix,lexicon_word_ngrams,lexicon_word_index,lexicon_thresholds_dict,lexicon,pred_values,tokens_to_lexicon_indices,freq_values):

    # dict holding all information computed for each fixation using the model outputs
    all_data = {}
    # is set to true when end of text is reached. For non-continuous reading, set to True when it reads the last trial
    end_of_task = False
    # the iterator that indicates the element of fixation in the text.
    # It goes backwards in case of regression (only continuous reading). For non-continuous reading, fixation = trial
    fixation = 0
    # the iterator that increases +1 with every next fixation/trial
    fixation_counter = 0
    # initialise attention window size
    attend_width = pm.attend_width
    # total number of tokens in input
    total_n_words = len(tokens)
    # history of regressions, set to true at a certain position in the text when a regression is performed to that word
    regression_flag = np.zeros(total_n_words, dtype=bool)
    # recognized word at position, which word received the highest activation in each position
    recognized_word_at_position = np.empty(total_n_words, dtype=object)
    # stores the amount of cycles needed for each word in text to be recognized
    recognized_word_at_cycle = np.zeros(total_n_words, dtype=int)
    recognized_word_at_cycle.fill(-1)
    # keep track of word activity in lexicon
    lexicon_word_activity = np.zeros((len(lexicon)), dtype=float)
    # keep track of changes in recognition threshold as the text is read, e.g. get effect of predictability
    lexicon_thresholds = np.zeros((len(lexicon)), dtype=float)
    # to keep track of positions in text whose thresholds have already been updated
    # to avoid updating it every time position is in stimulus
    updated_thresh_positions = []

    # dict storing info on saccade for each eye movement
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

    while not end_of_task:

        print(f'---Fixation {fixation_counter} at position {fixation}---')

        fixation_data = defaultdict(list)

        # make sure that fixation does not go over the end of the text. Needed for continuous reading
        fixation = min(fixation, len(tokens) - 1)

        fixation_data['foveal word'] = tokens[fixation]
        fixation_data['foveal word index'] = fixation
        fixation_data['attentional width'] = attend_width
        fixation_data['offset'] = saccade_info['offset_from_word_center']
        fixation_data['saccade_type'] = saccade_info['saccade_type']
        fixation_data['saccade error'] = saccade_info['saccade_error']
        fixation_data['saccade distance'] = saccade_info['saccade_distance']
        fixation_data['saccade_cause'] = saccade_info['saccade_cause']
        fixation_data['saccade_type_by_error'] = saccade_info['saccade_type_by_error']
        fixation_data['foveal word frequency'] = freq_values[tokens[fixation]] if tokens[fixation] in freq_values.keys() else 0
        fixation_data['foveal word predictability'] = pred_values[str(fixation)] if str(fixation) in pred_values.keys() else 0
        fixation_data['foveal word threshold'] = lexicon_thresholds[tokens_to_lexicon_indices[fixation]]

        # defining the stimulus
        stimulus, stimulus_position, fixated_position_in_stimulus = compute_stimulus(fixation, tokens)
        eye_position = compute_eye_position(stimulus, fixated_position_in_stimulus, saccade_info['offset_from_word_center'])
        fixation_data['stimulus'] = stimulus
        fixation_data['eye position'] = eye_position
        print(f"Stimulus: {stimulus}\n Eye position: {eye_position}")

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

        # define order to match activated words to slots in the stimulus
        # NV: the order list should reset when stimulus changes or with the first stimulus
        order_match_check = define_slot_matching_order(len(stimulus.split()), fixated_position_in_stimulus,
                                                       attend_width)
        print(f'Order for slot-matching: {order_match_check}')

        print('Entering cycle loops to define word activity...')
        print("fix on: " + tokens[fixation] + '  attent. width: ' + str(attend_width) + ' fixwrd thresh.' + str(round(lexicon_thresholds[tokens_to_lexicon_indices[fixation]],3)))
        shift = False
        n_cycles = 0
        n_cycles_since_attent_shift = 0
        attention_position = eye_position

        # define index of letters at the words edges.
        word_edges = find_word_edges(stimulus)

        # compute word input using ngram excitation and inhibition (out the cycle loop because this is constant)
        n_ngrams, total_ngram_activity, all_ngrams, word_input = compute_words_input(stimulus, lexicon_word_ngrams, eye_position, attention_position, attend_width, pm)
        fixation_data['n_ngrams'] = n_ngrams
        fixation_data['total_ngram_activity'] = total_ngram_activity
        # print("  input to fixwrd at first cycle: " + str(round(word_input[tokens_to_lexicon_indices[fixation]], 3)))

        # Counter n_cycles_since_attent_shift is 0 until attention shift (saccade program initiation),
        # then starts counting to 5 (because a saccade program takes 5 cycles, or 125ms.)
        while n_cycles_since_attent_shift < 5:

            # Update word act with word inhibition (input remains same, so does not have to be updated)
            lexicon_word_activity, lexicon_word_inhibition = update_word_activity(lexicon_word_activity, word_overlap_matrix, pm, word_input, all_ngrams, len(lexicon))

            # update cycle info
            foveal_word_index = lexicon_word_index[tokens[fixation]]
            foveal_word_activity = lexicon_word_activity[foveal_word_index]
            print('CYCLE ', str(n_cycles), '   activ @fix ', str(round(foveal_word_activity,3)))
            #print('        and act. of Die', str(round(lexicon_word_activity[lexicon_word_index[tokens[0]]],3)))
            fixation_data['foveal word activity per cycle'].append(foveal_word_activity)
            fixation_data['foveal word-to-word inhibition per cycle'].append(abs(lexicon_word_inhibition[foveal_word_index]))
            stim_activity = sum([lexicon_word_activity[lexicon_word_index[word]] for word in stimulus.split() if word in lexicon_word_index.keys()])
            fixation_data['stimulus activity per cycle'].append(stim_activity)
            total_activity = sum(lexicon_word_activity)
            fixation_data['lexicon activity per cycle'].append(total_activity)

            # check which words with activation above threshold
            above_thresh_lexicon = np.where(lexicon_word_activity > lexicon_thresholds, 1, 0)

            # word recognition, by checking matching active wrds to slots
            recognized_word_at_position, lexicon_word_activity = \
                match_active_words_to_input_slots(order_match_check,
                                                  stimulus,
                                                  recognized_word_at_position,
                                                  above_thresh_lexicon,
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

            # word selection and attention shift -> saccade decisions
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
            end_of_task = True
            print(recognized_word_at_position)
            print("END REACHED!")
            continue

        #if fixation_counter > 6: exit()
        # if end of text is not yet reached, compute next eye position and thus next fixation
        fixation, next_eye_position, saccade_info = compute_next_eye_position(pm, saccade_info, eye_position, stimulus, fixation, total_n_words, word_edges, fixated_position_in_stimulus)

    # register words in text in which no word in lexicon reaches recognition threshold
    unrecognized_words = dict()
    for position in range(total_n_words):
        if not recognized_word_at_position[position]:
            unrecognized_words[position] = tokens[position]

    return all_data, unrecognized_words

def word_recognition():
    pass
    # print('Defining stimulus...')
    # # stimulus in each fixation is pre-defined for controlled reading
    # stimulus = stim['all'][fixation]
    # eye_position = np.round(len(stimulus) // 2)
    # fixation_data['condition'] = stim['condition'][fixation]
    #
    # if pm.is_priming_task:
    #     prime = stim['prime'][fixation]
    #     fixation_data['prime'] = prime
    #
    # fixation_data['stimulus'] = stimulus
    # fixation_data['eye position'] = eye_position
    # print(f"stimulus: {stimulus}\nstart eye: {offset_from_word_center}\nfour letters right: {stimulus[eye_position:eye_position + 4]}")

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
        all_data, unrecognized_words = reading(pm,
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
        all_data, unrecognized_words = word_recognition()

    return all_data, unrecognized_words
