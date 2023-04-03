import numpy as np
from reading_helper_functions import string_to_open_ngrams, cal_ngram_exc_input, is_similar_word_length, \
    get_midword_position_for_surrounding_word, calc_word_attention_right, calc_saccade_error,\
    check_previous_refixations_at_position

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
    lexicon_word_inhibition = np.dot((overlap_select ** 2), -(lexicon_select ** 2))
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

def match_active_words_to_input_slots(order_match_check, stimulus, recognized_word_at_position, lexicon_thresholds, lexicon_word_activity, lexicon, min_activity, stimulus_position, len_sim_const):

    """
    Match active words to spatio-topic representation. Fill in the stops in the stimulus.
    The winner is the word with the highest activity above recognition threshold and of similar length.
    :return: recognized_word_at_position is the updated array of recognized words in each text position,
    lexicon_word_activity is the updated array with activity of each word in the lexicon
    """

    above_thresh_lexicon = np.where(lexicon_word_activity > lexicon_thresholds, 1, 0)

    for slot_to_check in range(len(order_match_check)):
        # slot_num is the slot in the stim (spot of still-unrecogn word) that we're checking
        slot_num = order_match_check[slot_to_check]
        word_index = slot_num
        # in continuous reading, recognized_word_at_position contains all words in text,
        # so word_index is the word position in the text (instead of in the stimulus)
        if stimulus_position:
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
            if int(np.sum(recognized_words_fit_len)):
                # Find the word with the highest activation in all words that have a similar length
                highest = np.argmax(recognized_words_fit_len * lexicon_word_activity)
                highest_word = lexicon[highest]
                print('word in input: ', word_searched, '      one w. highest act: ', highest_word)
                # The winner is matched to the slot,
                # and its activity is reset to minimum to not have it matched to other words
                recognized_word_at_position[word_index] = highest_word
                lexicon_word_activity[highest] = min_activity
                above_thresh_lexicon[highest] = 0

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
        saccade_info['saccade type'] = 'forward'
    elif next_fixation == 2: # AL: next fixation n + 2
        saccade_info['saccade type'] = 'wordskip'
        if regression_flag[fixation]:
            saccade_info['saccade cause'] = 2 # AL: bcs n resulted from regression and n + 1 has been recognized
        else:
            saccade_info['saccade cause'] = 1 # AL: bcs n + 2 has highest attwght (letter excitation)
    elif next_fixation == -1:
        saccade_info['saccade type'] = 'regression'
    elif next_fixation == 0:
        saccade_info['saccade type'] = 'refixation'
        if not recognized_word_at_position[fixation]:
            saccade_info['saccade cause'] = 1 # AL: bcs fixated word has not been recognized
        else:
            saccade_info['saccade_cause'] = 2 # AL: bcs right of fixated word has highest attwght (letter excitation)

    # AL: saccade distance is next attention position minus the current eye position
    if attention_position:
        saccade_info['saccade distance'] = attention_position - eye_position

    return attention_position, saccade_info

def compute_next_eye_position(pm, saccade_info, eye_position, stimulus, fixation, total_n_words, word_edges, fixated_position_in_stimulus, verbose=True):

    """
    This function computes next eye position and next offset from word center using saccade distance
    (defined by next attention position and current eye position) plus a saccade error.
    Importantly, it corrects the offset to prevent too short or too long saccades.
    :return: the next fixation, the next eye position and the updated saccade info
    """

    saccade_distance = saccade_info['saccade distance']
    offset_from_word_center = saccade_info['offset from word center']

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
        saccade_info['saccade type'] = 'regression'

    # Forward (include space between n and n+2)
    # AL: next eye position is too long (between space after last letter of fixated word and last letter of word n + 1)
    elif ((fixation < total_n_words - 1)
          and (next_eye_position > word_edges[fixated_position_in_stimulus][1])
          and (next_eye_position <= (word_edges[fixated_position_in_stimulus+1][1]))):
        # When saccade too short due to saccade error recalculate offset for n+1 (old offset is for N or N+2)
        if saccade_info['saccade type'] in ['wordskip','refixation']:
            center_position = get_midword_position_for_surrounding_word(1,word_edges,fixated_position_in_stimulus)
            offset_from_word_center = next_eye_position - center_position
            # AL: forward due to saccade error
            saccade_info['saccade type by error'] = True
        # Eye at (n+0 <-> n+1) space position
        if next_eye_position < word_edges[fixated_position_in_stimulus+1][0]:
            offset_from_word_center += 1
        fixation += 1
        saccade_info['saccade type'] = 'forward'

    # Wordskip
    # AL: next eye position after last letter of word n+1 and before 3 chars after last letter of word n+2
    elif ((fixation < total_n_words - 2)
          and (next_eye_position > word_edges[fixated_position_in_stimulus+1][1])
          and (next_eye_position <= word_edges[fixated_position_in_stimulus+2][1] + 2)):
        # AL: When saccade is too short, recalculate offset to correct eye position to n+2 (old offset is for n or n+1)
        if saccade_info['saccade type'] in ['forward','refixation']:
            # recalculate offset for n+2, todo check for errors
            center_position = get_midword_position_for_surrounding_word(2,word_edges,fixated_position_in_stimulus)
            offset_from_word_center = next_eye_position - center_position
            # AL: wordskip due to saccade error
            saccade_info['saccade type by error'] = True
        # Eye at (n+1 <-> n+2) space position
        if next_eye_position < word_edges[fixated_position_in_stimulus+2][0]:
            offset_from_word_center += 1
        # Eye at (> n+2) space position
        elif next_eye_position > word_edges[fixated_position_in_stimulus+2][1]:
            offset_from_word_center -= (next_eye_position - word_edges[fixated_position_in_stimulus+2][1])
        fixation += 2
        saccade_info['saccade type'] = 'wordskip'

    # Refixation due to saccade error
    # AL: weird to me, isn't this converting all saccades which were not refixation not due by error to refixation due by error?
    elif saccade_info['saccade type'] != 'refixation':
        # TODO find out if not regression is necessary
        center_position = np.round(word_edges[fixated_position_in_stimulus][0] +
                                  ((word_edges[fixated_position_in_stimulus][1] - word_edges[fixated_position_in_stimulus][0]) / 2.))
        offset_from_word_center = next_eye_position - center_position
        saccade_info['saccade type by error'] = True
        saccade_info['saccade type'] = 'refixation'

    # update saccade info
    saccade_info['offset from word center'] = float(offset_from_word_center)
    saccade_info['saccade distance'] = float(saccade_distance)
    saccade_info['saccade error'] = float(saccade_error)

    if verbose:
        if saccade_info['saccade type'] == 'forward':
            print(">->->->->->->->->->->->-")
        elif saccade_info['saccade type'] == 'wordskip':
            print(">>>>>>>>>>>>>>>>>>>>>>>>")
        elif saccade_info['saccade type'] == 'refixation':
            print("------------------------")
        elif saccade_info['saccade type'] == 'regression':
            print("<-<-<-<-<-<-<-<-<-<-<-<-")

    return fixation, next_eye_position, saccade_info