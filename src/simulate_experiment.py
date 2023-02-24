import logging
import numpy as np
import pickle
from collections import defaultdict
import math

from utils import get_word_freq, get_pred_values, check_previous_inhibition_matrix
from reading_functions import get_threshold, string_to_ngrams_and_locations, build_word_inhibition_matrix, \
    get_blankscreen_stimulus, calc_bigram_ext_input, calc_monogram_ext_input, define_slot_matching_order, is_similar_word_length, \
    sample_from_norm_distribution, find_word_edges, get_midword_position_for_surrounding_word, calc_word_attention_right, update_threshold, calc_saccade_error


logger = logging.getLogger(__name__)

def compute_stimulus(fixation, tokens):

    # assuming stimulus default is n-2 to n+2
    start_window = fixation - 2
    end_window = fixation + 2
    stimulus_position = [i for i in range(start_window,end_window+1) if (i >= 0 and i < len(tokens))] # only add position if after text begin and below text length
    stimulus = ' '.join([tokens[i] for i in stimulus_position])
    fixated_position_stimulus = stimulus_position.index(fixation)

    return stimulus, stimulus_position, fixated_position_stimulus

def compute_eye_position(stimulus, fixated_position_stimulus, offset_from_word_center):

    stimulus = stimulus.split(' ')
    center_of_fixation = round(len(stimulus[fixated_position_stimulus]) * 0.5)
    len_till_fix = sum([len(token)+1 for token in stimulus[:fixated_position_stimulus]]) # find length of stimulus (in characters) up until fixated word
    eye_position = len_till_fix + center_of_fixation + offset_from_word_center

    return int(np.round(eye_position))

def compute_ngram_activity(stimulus,lexicon_word_ngrams,eye_position,attention_position,attend_width,letPerDeg,attention_skew):

    unit_activations = {}
    all_ngrams = [ngram for word in stimulus.split(' ') for ngram in lexicon_word_ngrams[word][0]]
    all_ngram_locations = [lexicon_word_ngrams[word][1][ngram][0] for word in stimulus.split(' ') for ngram in lexicon_word_ngrams[word][0]]

    for ngram, location in zip(all_ngrams,all_ngram_locations):
        if len(ngram) == 2:
            activation = calc_bigram_ext_input(location,
                                                eye_position,
                                                attention_position,
                                                attend_width,
                                                letPerDeg,
                                                attention_skew)
        else:
            activation = calc_monogram_ext_input(location,
                                                eye_position,
                                                attention_position,
                                                attend_width,
                                                letPerDeg,
                                                attention_skew)

        if ngram in unit_activations.keys():
            unit_activations[ngram] = unit_activations[ngram] + activation
        else:
            unit_activations[ngram] = activation

    return unit_activations

def compute_words_activity(stimulus, lexicon_word_ngrams, eye_position, attention_position, attend_width, pm, fixation_data, word_overlap_matrix, tokens, fixation, lexicon_word_activity):

    lexicon_size = len(lexicon_word_ngrams.keys())
    lexicon_active_words = np.zeros((lexicon_size), dtype=bool)
    word_input = np.zeros((lexicon_size), dtype=float)
    crt_fixation_word_activities = dict()

    # define ngram activity given stimulus
    unit_activations = compute_ngram_activity(stimulus, lexicon_word_ngrams, eye_position,
                                              attention_position, attend_width, pm.letPerDeg,
                                              pm.attention_skew)
    fixation_data['ngram activity per cycle'].append(sum(unit_activations.values()))
    fixation_data['ngrams'].append(len(unit_activations.keys()))

    # compute word activity according to ngram excitation and inhibition
    # all stimulus bigrams used, therefore the same bigram inhibition for each word of lexicon (excit is specific to word, inhib same for all)
    # ngram_inhibition_input = sum(unit_activations.values()) + (pm.bigram_to_word_inhibition * len(unit_activations.keys()))
    ngram_inhibition_input = sum(unit_activations.values()) * pm.bigram_to_word_inhibition
    for lexicon_ix, lexicon_word in enumerate(lexicon_word_ngrams.keys()):
        word_excitation_input = 0
        # bigram & monogram activations
        ngram_intersect_list = set(unit_activations.keys()).intersection(set(lexicon_word_ngrams[lexicon_word][0]))
        for ngram in ngram_intersect_list:
            word_excitation_input += pm.bigram_to_word_excitation * unit_activations[ngram]
        word_input[lexicon_ix] = word_excitation_input + ngram_inhibition_input
        if lexicon_word == tokens[fixation]:
            crt_fixation_word_activities['word excitation'] = word_excitation_input
            crt_fixation_word_activities['ngram inhibition'] = (abs(ngram_inhibition_input))

    # normalize based on number of ngrams in lexicon
    all_ngrams = list()
    for info_tuple in lexicon_word_ngrams.values():
        all_ngrams.append(len(info_tuple[0]))
    word_input = word_input / np.array(all_ngrams)

    # re-compute word activity using to word-to-word inhibition
    # NV: the more active a certain word is, the more inhibition it will execute on its peers -> activity is multiplied by inhibition constant.
    # NV: then, this inhibition value is weighed by how much overlap there is between that word and every other.
    lexicon_normalized_word_inhibition = (100.0/lexicon_size) * pm.word_inhibition
    # find which words are active
    lexicon_active_words[(lexicon_word_activity > 0.0) | (word_input > 0.0)] = True
    overlap_select = word_overlap_matrix[:, (lexicon_active_words == True)]
    lexicon_select = (lexicon_word_activity + word_input)[(lexicon_active_words == True)] * lexicon_normalized_word_inhibition
    # This concentrates inhibition on the words that have most overlap and are most active
    lexicon_word_inhibition = np.dot((overlap_select ** 2), -(lexicon_select ** 2)) / np.array(len(set(all_ngrams)))
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

    return lexicon_word_activity, crt_fixation_word_activities, lexicon_word_inhibition

def match_active_words_to_input_slots(order_match_check, stimulus, recognized_position_flag, recognized_word_at_position_flag, recognized_word_at_position, above_thresh_lexicon, lexicon_word_activity, lexicon, min_activity, stimulus_position, len_sim_const):

    n_words_in_stim = len(stimulus.split())
    new_recognized_words = np.zeros(len(lexicon))

    for slot_to_check in range(n_words_in_stim):
        # slot_num is the slot in the stim (spot of still-unrecogn word) that we're checking
        slot_num = order_match_check[slot_to_check]
        word_index = stimulus_position[slot_num]
        # if the slot has not yet been filled
        if not recognized_position_flag[word_index]:
            # Check words that have the same length as word in the slot we're now looking for
            word_searched = stimulus.split()[slot_num]
            # MM: recognWrdsFittingLen_np: array with 1=wrd act above threshold, & approx same len
            # as to-be-recogn wrd (with 'len_sim_const' margin), 0=otherwise
            recognized_words_fit_len = above_thresh_lexicon * np.array([int(is_similar_word_length(len(x.replace('_', '')),len(word_searched),len_sim_const)) for x in lexicon])
            # if at least one word matches (act above threshold and similar length)
            if sum(recognized_words_fit_len):
                # Find the word with the highest activation in all words that have a similar length
                highest = np.argmax(recognized_words_fit_len * lexicon_word_activity)
                highest_word = lexicon[highest]
                print('word in input: ', word_searched)
                print('word with highest activation: ', highest_word)
                # The winner is matched to the slot, and its activity is reset to minimum to not have it matched to other words
                recognized_position_flag[word_index] = True
                recognized_word_at_position[word_index] = highest_word
                above_thresh_lexicon[highest] = 0
                lexicon_word_activity[highest] = min_activity
                new_recognized_words[highest] = 1
                # if recognized word equals word in the stimulus
                if highest_word == word_searched:
                    recognized_word_at_position_flag[word_index] = True

    return recognized_position_flag, recognized_word_at_position, recognized_word_at_position_flag, lexicon_word_activity, new_recognized_words

def compute_next_attention_position(all_data,tokens,fixation,word_edges,fixated_position_in_stimulus,fixation_first_position_to_eye,regression_flag,recognized_position_flag,lexicon_word_activity,eye_position,fixation_counter,attention_position,attend_width, fix_lexicon_index, pm):

    # Define next fixation (n, n-1, n+1 or n+2). Default next fixation is n+1
    next_fixation = 1
    word_reminder_length = word_edges[fixated_position_in_stimulus][1] - eye_position
    refix_size = pm.refix_size

    # regression: if the current fixation was a regression and next word has been recognized, move eyes to n+2 to resume reading
    if regression_flag[fixation] and recognized_position_flag[fixation + 1]:
        next_fixation = 2

    # regression: check whether the previous word was recognized or there was already a regression performed. If not: regress
    elif fixation > 1 and not recognized_position_flag[fixation - 1] and not regression_flag[fixation - 1]:
        next_fixation = -1

    # refixation: refixate if the foveal word is not recognized but is still being processed.
    elif (not recognized_position_flag[fixation]) and (lexicon_word_activity[fix_lexicon_index] > 0):
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

    # Calculate next attention position based on next fixation estimate
    if next_fixation == 0:
        # MM: if we're refixating same word because it has highest attentwgt AL: or not being recognized whilst processed
        # ...use first refixation middle of remaining half as refixation stepsize
        attention_position = fixation_first_position_to_eye[1] + refix_size
    else:
        assert (next_fixation in [-1, 1, 2])
        attention_position = get_midword_position_for_surrounding_word(next_fixation, word_edges, fixated_position_in_stimulus)

    return attention_position, next_fixation

def compute_next_fixation(pm, saccade_distance, offset_from_word_center, eye_position, stimulus, center_word_first_letter_index, center_word_last_letter_index, left_word_edge_letter_indices, right_word_edge_letter_indices, fixation, total_n_words, regression, refixation, refixation_type, wordskip, forward):

    # normal random error based on difference with optimal saccade distance
    saccade_error = calc_saccade_error(saccade_distance,
                                       pm.sacc_optimal_distance,
                                       pm.saccErr_scaler,
                                       pm.saccErr_sigma,
                                       pm.saccErr_sigma_scaler,
                                       pm.use_saccade_error)
    saccade_distance = saccade_distance + saccade_error
    offset_from_word_center = offset_from_word_center + saccade_error

    # compute the position of the next fixation
    next_eye_position = int(np.round(eye_position + saccade_distance))
    if next_eye_position >= len(stimulus) - 1:
        next_eye_position = len(stimulus) - 2

    # Calculating the actual saccade type
    saccade_type_by_error = 0  # 0: no error, 1: refixation, 2: forward, 3: wordskip by error

    # Regressions
    if next_eye_position < center_word_first_letter_index:
        # eye at right space position
        if next_eye_position > left_word_edge_letter_indices[-2][1]:
            offset_from_word_center -= 1
        if next_eye_position < left_word_edge_letter_indices[-2][0]:
            centerposition_r = get_midword_position_for_surrounding_word(-1,right_word_edge_letter_indices,left_word_edge_letter_indices)
            offset_from_word_center = centerposition_r - left_word_edge_letter_indices[-2][0]
        fixation -= 1
        regression = True
        refixation, wordskip, forward = False, False, False
        print("<-<-<-<-<-<-<-<-<-<-<-<-")

    # Forward (include space between n and n+2)
    elif ((fixation < total_n_words - 1)
          and (next_eye_position > center_word_last_letter_index)
          and (next_eye_position <= (right_word_edge_letter_indices[1][1]))):
        # When saccade too short due to saccade error recalculate offset for n+1 (old offset is for N or N+2)
        if wordskip or refixation:
            center_position = get_midword_position_for_surrounding_word(1,right_word_edge_letter_indices,left_word_edge_letter_indices)
            offset_from_word_center = next_eye_position - center_position
            saccade_type_by_error = 2
        # Eye at (n+0 <-> n+1) space position
        if next_eye_position < right_word_edge_letter_indices[1][0]:
            offset_from_word_center += 1
        fixation += 1
        forward = True
        regression, refixation, wordskip = False, False, False
        print(">->->->->->->->->->->->-")

    # Wordskip
    elif ((fixation < total_n_words - 2)
          and (next_eye_position > right_word_edge_letter_indices[1][1])
          and (next_eye_position <= right_word_edge_letter_indices[2][1] + 2)):
        if forward or refixation:
            # recalculate offset for n+2, todo check for errors
            center_position = get_midword_position_for_surrounding_word(2,right_word_edge_letter_indices,left_word_edge_letter_indices)
            offset_from_word_center = next_eye_position - center_position
            saccade_type_by_error = 3
        # Eye at (n+1 <-> n+2) space position
        if next_eye_position < right_word_edge_letter_indices[2][0]:
            offset_from_word_center += 1
        # Eye at (> n+2) space position
        elif next_eye_position > right_word_edge_letter_indices[2][1]:
            offset_from_word_center -= (next_eye_position - right_word_edge_letter_indices[2][1])
        fixation += 2
        wordskip = True
        regression, refixation, forward = False, False, False
        print(">>>>>>>>>>>>>>>>>>>>>>>>")

    # Refixation
    else:
        # Refixation due to saccade error
        if not refixation:
            # TODO find out if not regression is necessary
            center_position = np.round(center_word_first_letter_index +
                                      ((center_word_last_letter_index - center_word_first_letter_index) / 2.))
            offset_from_word_center = next_eye_position - center_position
            saccade_type_by_error = 1
            refixation_type = 3
        refixation = True
        regression, wordskip, forward = False, False, False
        print("------------------------")

    return saccade_distance, saccade_error, offset_from_word_center, fixation, regression, refixation, refixation_type, wordskip, forward, next_eye_position, saccade_type_by_error

def continuous_reading(pm,tokens,word_overlap_matrix,lexicon_word_ngrams,lexicon_word_index,total_n_words,lexicon_thresholds_dict,lexicon,pred_values,tokens_to_lexicon_indices,freq_values):

    all_data = {}
    # is set to true when end of text is reached. For non-continuous reading, set to True when it reads the last trial
    end_of_task = False
    # the iterator that indicates the element of fixation in the text. It goes backwards in case of regression (only continuous reading). For non-continuous reading, fixation = trial
    fixation = 0
    # the iterator that increases +1 with every next fixation/trial
    fixation_counter = 0
    # initialise attention window size
    attend_width = pm.attend_width
    # if eye position is to be in a position other than that of the word middle, offset will be negative/positive (left/right) and will represent the number of letters to the new position. It's value is reset before a new saccade is performed.
    offset_from_word_center = 0
    # max pred to compute new threshold based on pred also (not only frequency)
    max_predictability = max(pred_values.values())
    regression, wordskip, refixation, forward = False, False, False, False
    saccade_distance, saccade_error, refixation_type, wordskip_pass, saccade_type_by_error, offset_previous = 0, 0, 0, 0, 0, 0
    # history of regressions, is set to true at a certain position in the text when a regression is performed to that word
    regression_flag = np.zeros(total_n_words, dtype=bool)
    # recognition flag for each word position in the text is set to true when a word whose length is similar to that of the fixated word, is recognised so if it fulfills the condition is_similar_word_length(fixated_word,other_word)
    recognized_position_flag = np.zeros(total_n_words, dtype=bool)
    # recognition flag for each word in the text, it is set to true whenever the exact word from the stimuli is recognized
    recognized_true_word_flag = np.zeros(total_n_words, dtype=bool)
    # recognized word at position, which word received the highest activation in each position
    recognized_word_at_position = np.empty(total_n_words, dtype=object)
    # stores the amount of cycles needed for each word in text to be recognized
    recognized_word_at_cycle = np.zeros(total_n_words, dtype=int)
    recognized_word_at_cycle.fill(-1)
    # keep track of word activity in lexicon
    lexicon_word_activity = np.zeros((len(lexicon)), dtype=float)
    # to keep track of changes in recognition threshold as the text is read, e.g. get effect of predictability
    lexicon_thresholds = np.zeros((len(lexicon)), dtype=float)
    # to keep track of positions in text whose thresholds have already been updated to avoid updating it every time position is in stimulus
    updated_thresh_positions = []

    # initialize thresholds with values based frequency, if dict has been filled (if frequency flag)
    if lexicon_thresholds_dict != {}:
        for i, word in enumerate(lexicon):
            lexicon_thresholds[i] = lexicon_thresholds_dict[word]

    while not end_of_task:

        print(f'---Fixation {fixation_counter+1} at position {fixation+1}---')

        fixation_data = defaultdict(list)

        # make sure that fixation does not go over the end of the text. Needed for continuous reading
        fixation = min(fixation, len(tokens) - 1)

        fixation_data['foveal word'] = tokens[fixation]
        fixation_data['foveal word index'] = fixation
        fixation_data['attentional width'] = attend_width
        fixation_data['offset'] = offset_from_word_center
        fixation_data['regressed'] = regression
        fixation_data['refixated'] = refixation
        fixation_data['wordskip'] = wordskip
        fixation_data['forward'] = forward
        fixation_data['relative landing position'] = offset_previous
        fixation_data['saccade error'] = saccade_error
        fixation_data['saccade distance'] = int(np.round(saccade_distance))
        fixation_data['wordskip pass'] = wordskip_pass
        fixation_data['refixation type'] = refixation_type
        fixation_data['saccade_type_by_error'] = saccade_type_by_error
        fixation_data['word frequency'] = 0
        fixation_data['word predictability'] = 0
        fixation_data['word threshold'] = lexicon_thresholds[tokens_to_lexicon_indices[fixation]]

        print('Defining stimulus...')
        stimulus, stimulus_position, fixated_position_in_stimulus = compute_stimulus(fixation, tokens)
        eye_position = compute_eye_position(stimulus, fixated_position_in_stimulus, offset_from_word_center)
        fixation_data['stimulus'] = stimulus
        fixation_data['eye position'] = eye_position
        print(
            f"stimulus: {stimulus}\neye position in stimulus: {eye_position}\nfour characters to the right of fixation: {stimulus[eye_position:eye_position + 4]}")

        # define order to match activated words to slots in the stimulus
        # NV: the order list should reset when stimulus changes or with the first stimulus
        order_match_check = define_slot_matching_order(len(stimulus.split()),fixated_position_in_stimulus)

        # define attention width according to whether there was a regression in the last fixation, i.e. this fixation location is a result of regression
        if regression:
            # set regression flag to know that a regression has been realized towards this position, in order to prevent double regressions to the same word
            regression_flag[fixation] = True
            # in principle, next saccade should not be a regression again
            regression = False
            # narrow attention width by 2 letters in the case of regressions
            attend_width = max(attend_width - 2.0, pm.min_attend_width)
        else:
            # widen attention by 0.5 letters in forward saccades
            attend_width = min(attend_width + 0.5, pm.max_attend_width)

        print('Entering cycle loops to define word activity...')
        shift = False
        print(("fix on: " + tokens[fixation] + '  attention width: ' + str(attend_width)))
        amount_of_cycles = 0
        amount_of_cycles_since_attention_shifted = 0
        attention_position = eye_position
        fixation_center = eye_position - int(np.round(offset_from_word_center))

        # define edge locations of words, as well as location of first letter to the right and first letter to the left from center of fixated word
        # right_word_edge_letter_indices, left_word_edge_letter_indices, fixation_first_position_right_to_middle, fixation_first_position_left_to_middle, fixated_word_edge_indices = \
        #     find_word_edges(fixation_center, eye_position, stimulus, tokens)
        word_edges = find_word_edges(stimulus)

        fixation_first_position_left_to_eye = eye_position - 1 if eye_position - 1 > 0 else eye_position
        fixation_first_position_right_to_eye = eye_position + 1 if eye_position + 1 < len(tokens) else eye_position
        fixation_first_position_to_eye = (fixation_first_position_left_to_eye, fixation_first_position_right_to_eye)

        # update lexeme thresholds with predictability values (because one word form may have different pred values depending on context
        # AL: changed placed, now it updates thresholds with pred only for next word and depending on recognition
        # if pm.prediction_flag:
        #     positions = [position for position in stimulus_position]
        #     for position_in_text in positions:
        #         if position_in_text not in updated_thresh_positions:
        #             position_in_lexicon = tokens_to_lexicon_indices[position_in_text]
        #             lexicon_thresholds[position_in_lexicon] = update_threshold(position_in_text,
        #                                                                        lexicon_thresholds[position_in_lexicon],
        #                                                                        max_predictability,
        #                                                                        pm.wordpred_p,
        #                                                                        pred_values)
        #             updated_thresh_positions.append(position_in_text)
        # fixation_data['word threshold'] = lexicon_thresholds[tokens_to_lexicon_indices[fixation]]

        # A saccade program takes 5 cycles, or 125ms. This counter starts counting at saccade program initiation.
        while amount_of_cycles_since_attention_shifted < 5:

            print(f'CYCLE {amount_of_cycles}')

            # define word activity in lexicon
            lexicon_word_activity, crt_fixation_word_activities, lexicon_word_inhibition = compute_words_activity(stimulus, lexicon_word_ngrams, eye_position,
                                                                                                                  attention_position, attend_width, pm, fixation_data, word_overlap_matrix,
                                                                                                                  tokens, fixation, lexicon_word_activity)

            foveal_word_index = lexicon_word_index[tokens[fixation]]
            foveal_word_activity = lexicon_word_activity[foveal_word_index]
            fixation_data['foveal word activity per cycle'].append(foveal_word_activity)
            crt_fixation_word_activities['between word inhibition'] = abs(lexicon_word_inhibition[foveal_word_index])
            crt_fixation_word_activities['foveal word activity'] = foveal_word_activity  # activity of foveal word in last cycle before shift
            stim_activity = sum([lexicon_word_activity[lexicon_word_index[word]] for word in stimulus.split() if word in lexicon_word_index.keys()])
            fixation_data['stimulus activity per cycle'].append(stim_activity)
            total_activity = sum(lexicon_word_activity)
            fixation_data['lexicon activity per cycle'].append(total_activity)
            crt_fixation_word_activities['total activity'] = total_activity  # total activity of lexicon before shift

            # check which words with activation above threshold
            above_thresh_lexicon = np.where(lexicon_word_activity > lexicon_thresholds, 1, 0)
            # fixation_data['exact recognized words positions'].append([i for i in above_thresh_lexicon if i == 1])
            # fixation_data['exact recognized words'].append([lexicon[i] for i in above_thresh_lexicon if i == 1])

            # word recognition, by checking matching active wrds to slots
            recognized_position_flag, recognized_word_at_position, recognized_true_word_flag, lexicon_word_activity, new_recognized_words = \
                match_active_words_to_input_slots(order_match_check,
                                                  stimulus,
                                                  recognized_position_flag,
                                                  recognized_true_word_flag,
                                                  recognized_word_at_position,
                                                  above_thresh_lexicon,
                                                  lexicon_word_activity,
                                                  lexicon,
                                                  pm.min_activity,
                                                  stimulus_position,
                                                  pm.word_length_similarity_constant)

            # update threshold of n+1 or n+2 with pred value
            if pm.prediction_flag:
                # update the threshold for the next word only if word at fixation has been recognized
                if recognized_true_word_flag[fixation]:
                    position = fixation + 1
                    # if next word has already been recognized, update threshold of n+2 word
                    if recognized_true_word_flag[position]:
                        position = fixation + 2
                    # and only if next word has not been recognized (either n+1 or n+2)
                    if not recognized_true_word_flag[position]:
                        # and if it has not been updated yet
                        if position not in updated_thresh_positions:
                            position_in_lexicon = tokens_to_lexicon_indices[position]
                            lexicon_thresholds[position_in_lexicon] = update_threshold(position_in_lexicon,
                                                                                       lexicon_thresholds[position_in_lexicon],
                                                                                       max_predictability,
                                                                                       pm.wordpred_p,
                                                                                       pred_values)
                            updated_thresh_positions.append(position)

            # word selection and attention shift -> saccade decisions
            if not shift:
                # MM: on every cycle, take sample (called shift_start) out of normal distrib.
                # If cycle since fixstart > sample, make attentshift. This produces approx ex-gauss SRT
                if recognized_position_flag[fixation]:
                    # MM: if word recog, then faster switch (norm. distrib. with <mu) than if not recog.
                    shift_start = sample_from_norm_distribution(pm.mu, pm.sigma, pm.distribution_param,recognized=True)
                else:
                    shift_start = sample_from_norm_distribution(pm.mu, pm.sigma, pm.distribution_param,recognized=False)
                # shift attention (do a saccade) if amount of cycles has reached shift start
                if amount_of_cycles >= shift_start:
                    shift = True
                    offset_from_word_center = 0
                    attention_position, next_fixation = compute_next_attention_position(all_data,
                                                                                        tokens,
                                                                                        fixation,
                                                                                        word_edges,
                                                                                        fixated_position_in_stimulus,
                                                                                        fixation_first_position_to_eye,
                                                                                        regression_flag,
                                                                                        recognized_position_flag,
                                                                                        lexicon_word_activity,
                                                                                        eye_position,
                                                                                        fixation_counter,
                                                                                        attention_position,
                                                                                        attend_width,
                                                                                        foveal_word_index,
                                                                                        pm)
                    if next_fixation == 1: # next fixation n + 1
                        forward = True
                    elif next_fixation == 2: # next fixation n + 2
                        wordskip = True
                        if regression_flag[fixation]:
                            wordskip_pass = 2 # bcs n resulted from regression and n + 1 has been recognized
                        else:
                            wordskip_pass = 1 # bcs n + 2 has highest attwght (letter excitation)
                    elif next_fixation == -1:
                        regression = True
                    elif next_fixation == 0:
                        refixation = True
                        if not recognized_position_flag[fixation]:
                            refixation_type = 1 # bcs fixated word has not been recognized
                        else:
                            refixation_type = 2 # bcs right of fixated word has highest attwght (letter excitation)

                    saccade_distance = attention_position - eye_position

            if shift:
                if amount_of_cycles_since_attention_shifted < 1:
                    crt_fixation_word_activities_at_shift = crt_fixation_word_activities
                # count the amount of cycles since attention shift
                amount_of_cycles_since_attention_shifted += 1

            if recognized_position_flag[fixation] and recognized_word_at_cycle[fixation] == -1:
                # MM: here the time to recognize the word gets stored
                recognized_word_at_cycle[fixation] = amount_of_cycles
                fixation_data['recognition cycle'] = recognized_word_at_cycle[fixation]

            attention_position = np.round(attention_position)
            amount_of_cycles += 1

        # out of cycle loop. After last cycle, compute fixation duration and add final values for fixated word before shift is made
        fixation_duration = amount_of_cycles * pm.cycle_size
        fixation_data['fixation duration'] = fixation_duration
        fixation_data['recognized word at position'] = recognized_word_at_position
        fixation_data['word excitation'] = crt_fixation_word_activities_at_shift['word excitation']
        fixation_data['ngram inhibition'] = crt_fixation_word_activities_at_shift['ngram inhibition']
        fixation_data['between word inhibition'] = crt_fixation_word_activities_at_shift['between word inhibition']
        fixation_data['word activity'] = crt_fixation_word_activities_at_shift['foveal word activity']
        fixation_data['total activity'] = crt_fixation_word_activities_at_shift['total activity']
        if tokens[fixation] in freq_values.keys():
            fixation_data['word frequency'] = freq_values[tokens[fixation]]
        if str(fixation) in pred_values.keys():
            fixation_data['word predictability'] = pred_values[str(fixation)]

        all_data[fixation_counter] = fixation_data
        print(recognized_word_at_position)

        word_act = fixation_data['word activity']
        if 'recognition cycle' in fixation_data.keys():
            word_act = fixation_data['foveal word activity per cycle'][fixation_data['recognition cycle']]
        print(f"Relative activity from foveal word: { word_act / fixation_data['word threshold']}")

        print("Fixation duration: ", fixation_data['fixation duration'], " ms.")
        if recognized_position_flag[fixation]:
            if recognized_true_word_flag[fixation]:
                print("The correct word was recognized at fixation position!")
            else:
                print("Another word was recognized at fixation position!")
                print(f"Recognized word: {recognized_word_at_position[fixation]}")
        else:
            print("No word was recognized at fixation position")

        fixation_counter += 1

        # Check if end of text is reached
        if fixation == total_n_words - 1:
            end_of_task = True
            print("END REACHED!")
            continue
        print(fixation_data)
        exit()
        # if end of text is not yet reached, compute some values for next fixation
        saccade_distance, saccade_error, offset_from_word_center, fixation, regression, refixation, refixation_type, wordskip, forward, next_eye_position, saccade_type_by_error = \
            compute_next_fixation(pm, saccade_distance, offset_from_word_center, eye_position, stimulus,
                                  center_word_first_letter_index, center_word_last_letter_index,
                                  left_word_edge_letter_indices, right_word_edge_letter_indices, fixation,
                                  total_n_words, regression, refixation, refixation_type, wordskip, forward)

        # check if next fixation will be at the last word of the text and stop if next eye position is at (last letter -1) of text to prevent errors
        if fixation == total_n_words - 1 and next_eye_position >= len(stimulus) - 3:
            end_of_task = True
            print("END REACHED!")
            continue
        # print(fixation_data)
    # register words in text in which no word in lexicon reaches recognition threshold
    unrecognized_words = dict()
    for position in range(total_n_words):
        if not recognized_position_flag[position]:
            unrecognized_words[position] = tokens[position]

    return all_data, unrecognized_words

def controlled_reading():
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

    # TODO adapt code to add affix system
    # TODO adapt code to add grammar
    # TODO add visualise_reading

    print('Preparing simulation...')

    if type(pm.stim_all) == str:
        tokens = pm.stim_all.split(' ')
    else:
        tokens = [token for stimulus in pm.stim_all for token in stimulus.split(' ')]

    if pm.is_priming_task:
        tokens.extend([token for stimulus in list(pm.stim["prime"]) for token in stimulus.split(' ')])

    tokens = [word.strip() for word in tokens if word.strip() != '']

    cleaned_words = [token.replace(".", "").lower() for token in set(tokens)]
    word_frequencies = get_word_freq(pm,cleaned_words)
    pred_values = get_pred_values(pm,cleaned_words)
    max_frequency = max(word_frequencies.values())
    lexicon = list(set(tokens) | set(word_frequencies.keys())) # it is actually just the words in the input, because word_frequencies is the overlap between words in the freq resource

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
        word_local = " " + word + " "
        all_word_ngrams, word_ngram_locations = string_to_ngrams_and_locations(word_local,pm)
        lexicon_word_ngrams[word] = (all_word_ngrams, word_ngram_locations)
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
        all_data, unrecognized_words = continuous_reading(pm,
                                                        tokens,
                                                        word_inhibition_matrix,
                                                        lexicon_word_ngrams,
                                                        lexicon_word_index,
                                                        total_n_words,
                                                        word_thresh_dict,
                                                        lexicon,
                                                        pred_values,
                                                        tokens_to_lexicon_indices,
                                                        word_frequencies)

    else:
        all_data, unrecognized_words = controlled_reading()

    return all_data, unrecognized_words







