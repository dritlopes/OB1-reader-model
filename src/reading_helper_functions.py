import numpy as np
import pickle
import os
import math
import re

def get_stimulus_edge_positions(stimulus):

    # AL: get position of letters next to spaces
    stimulus_word_edge_positions = []
    for letter_position in range(1,len(stimulus)-1):
        if stimulus[letter_position+1] == " " or stimulus[letter_position-1] == " ":
            stimulus_word_edge_positions.append(letter_position-1)

    return stimulus_word_edge_positions

def string_to_open_ngrams(string,gap):

    all_ngrams, all_weights, all_locations = [], [], []

    # AL: make sure string contains one space before and after for correctly finding word edges
    string = " " + string + " "
    edge_locations = get_stimulus_edge_positions(string)
    string = string.strip()

    for position, letter in enumerate(string):
        weight = 0.5
        # AL: to avoid ngrams made of spaces
        if letter != ' ':
            if position in edge_locations:
                weight = 1.
                # AL: increases weigth of unigrams with no crowding (one-letter words, e.g. "a")
                if position > 0 and position < len(string) - 1:
                    if string[position-1] == ' ' and string[position+1] == ' ':
                        weight = 2.0
                # AL: include monogram if at word edge
                all_ngrams.append(letter)
                all_weights.append(weight)
                all_locations.append([position])

            # AL: find bigrams
            for i in range(1, gap+1):
                # AL: make sure second letter of bigram does not cross the stimulus string nor the word
                if position+i >= len(string) or string[position+i] == ' ':
                    break
                # Check if second letter in bigram is edge
                if position+i in edge_locations:
                    weight = weight * 2
                bigram = letter+string[position+i]
                all_ngrams.append(bigram)
                all_locations.append([position,position+i])
                all_weights.append(weight)

    return all_ngrams, all_weights, all_locations

def normalize_values(p, values, max_value):

    return ((p * max_value) - values) / (p * max_value)

def get_threshold(word, word_freq_dict, max_frequency, freq_p, max_threshold):

    # should always ensure that the maximum possible value of the threshold doesn't exceed the maximum allowable word activity
    # let threshold be fun of word freq. freq_p weighs how strongly freq is (1=max, then thresh. 0 for most freq. word; <1 means less havy weighting)
    # from 0-1, inverse of frequency, scaled to 0(highest freq)-1(lowest freq)
    word_threshold = max_threshold
    try:
        word_frequency = word_freq_dict[word]
        word_threshold = word_threshold * ((max_frequency/freq_p) - word_frequency) / (max_frequency/freq_p)
        # AL: changed this to be like in paper
        # word_threshold = 0.22 * ((max_frequency/freq_p) - word_frequency) / (max_frequency/freq_p)
        # AL: alter the size of effect of frequency and pred as a function of length effect on threshold, as in paper
        # word_threshold = word_threshold * (1 - .61**(-0.44*len(word)))
    except KeyError:
        print(f'Word {word} not in frequency map')

    return word_threshold

def update_threshold(word_position, word, word_threshold, max_predictability, pred_p, pred_values):

    word_pred = pred_values[str(word_position)]['predictions'][word]
    # word_pred = normalize_values(pred_p,float(word_pred),max_predictability)
    word_threshold = word_threshold * ((max_predictability/pred_p) - word_pred) / (max_predictability/pred_p)

    return word_threshold

def update_lexicon_threshold(recognized_word_at_position,fixation,tokens,updated_thresh_positions,lexicon_thresholds,wordpred_p,pred_values,tokens_to_lexicon_indices,lexicon):

    # # AL: update threshold of each predicted word
    # position = check_predictability(recognized_word_at_position, fixation, tokens, updated_thresh_positions)
    # if position:
    #     predicted_words = list(pred_values[str(position)]['predictions'].keys())
    #     for predicted_word in predicted_words:
    #         if predicted_word in lexicon:
    #             position_in_lexicon = lexicon.index(predicted_word)
    #             lexicon_thresholds[position_in_lexicon] = update_threshold(position,
    #                                                                        predicted_word,
    #                                                                        lexicon_thresholds[position_in_lexicon],
    #                                                                        max(pred_values.values()),
    #                                                                        wordpred_p,
    #                                                                        pred_values)
    #     updated_thresh_positions.append(position)

    # AL: update threshold just from word in text
    position = check_predictability(recognized_word_at_position, fixation, tokens, updated_thresh_positions)
    if position:
        predicted_words = list(pred_values[str(position)]['predictions'].keys())
        if tokens[position] in predicted_words:
            position_in_lexicon = tokens_to_lexicon_indices[position]
            lexicon_thresholds[position_in_lexicon] = update_threshold(position,
                                                                       tokens[position],
                                                                       lexicon_thresholds[position_in_lexicon],
                                                                       max(pred_values[str(position)]['predictions'].values()),
                                                                       wordpred_p,
                                                                       pred_values)
        updated_thresh_positions.append(position)

    return updated_thresh_positions, lexicon_thresholds

def check_predictability(recognized_word_at_position, fixation, tokens, updated_positions):

    position = None
    # update next word only if word at fixation has been recognized correctly
    if recognized_word_at_position[fixation] == tokens[fixation]:
        # and next word has not been recognized yet, nor has been already updated
        if not recognized_word_at_position[fixation + 1] and fixation + 1 not in updated_positions:
            position = fixation + 1
        # if n+1 has already been recognized or updated, update n+2 word if it exists and it has not been updated yet
        elif fixation < len(tokens) - 2 and recognized_word_at_position[fixation + 1] == tokens[fixation + 1] and fixation + 2 not in updated_positions:
            position = fixation + 2

    return position

def is_similar_word_length(len1, len2, len_sim_constant):

    is_similar = False
    # NV: difference of word length  must be within 15% of the length of the longest word
    if abs(len1-len2) < (len_sim_constant * max(len1, len2)):
        is_similar = True

    return is_similar

def build_word_inhibition_matrix(lexicon,lexicon_word_ngrams,pm,matrix_filepath,matrix_parameters_filepath):

    lexicon_size = len(lexicon)
    word_overlap_matrix = np.zeros((lexicon_size, lexicon_size), dtype=float)
    # word_inhibition_matrix = np.empty((lexicon_size, lexicon_size), dtype=bool)

    for word_1_index in range(lexicon_size):    # MM: receiving unit, I think...
        # AL: make sure word1-word2, but not word2-word1 or word1-word1.
        for word_2_index in range(word_1_index+1,lexicon_size):    # MM: sending unit, I think...
            word1, word2 = lexicon[word_1_index], lexicon[word_2_index]
            length_sim = 1 - (abs(len(word1)-len(word2))/max(len(word1),len(word2)))
            # if not is_similar_word_length(len(word1), len(word2), pm.word_length_similarity_constant):
            #     continue
            # else:
            # AL: lexicon_word_ngrams already contains all ngrams (bigrams and included monograms)
            ngram_common = list(set(lexicon_word_ngrams[word1]).intersection(set(lexicon_word_ngrams[word2])))
            n_total_overlap = len(ngram_common)
            # MM: now inhib set as proportion of overlapping bigrams (instead of nr overlap)
            word_overlap_matrix[word_1_index, word_2_index] = (n_total_overlap / (len(lexicon_word_ngrams[word1]))) * length_sim
            word_overlap_matrix[word_2_index, word_1_index] = (n_total_overlap / (len(lexicon_word_ngrams[word2]))) * length_sim
            #print("word1 ", word1, "word2 ", word2, "overlap ", n_total_overlap, "len w1 ", len(lexicon_word_ngrams[word1]))
            #print("inhib one way", word_overlap_matrix[word_1_index, word_2_index])

    with open(matrix_filepath, "wb") as f:
        pickle.dump(word_overlap_matrix, f)

    size_of_file = os.path.getsize(matrix_filepath)
    with open(matrix_parameters_filepath, "wb") as f:
        pickle.dump(str(lexicon_word_ngrams) + str(lexicon_size) +
                    str(pm.simil_algo) + str(pm.max_edit_dist) + str(pm.short_word_cutoff) + str(size_of_file), f)

    return word_overlap_matrix

def get_blankscreen_stimulus(blankscreen_type):

    if blankscreen_type == 'blank':  # NV decide what type of blank screen to show
        stimulus = ""

    elif blankscreen_type == 'hashgrid':
        stimulus = "#####"  # NV: overwrite stimulus with hash grid

    elif blankscreen_type == 'fixation cross':
        stimulus = "+"

    return stimulus

def get_attention_skewed(attentionWidth, attention_eccentricity, attention_skew):

    # Remember to remove the abs with calc functions
    if attention_eccentricity < 0:
        # Attention left
        attention = 1.0/(attentionWidth)*math.exp(-(pow(abs(attention_eccentricity), 2)) /
                                                  (2*pow(attentionWidth/attention_skew, 2))) + 0.25
    else:
        # Attention right
        attention = 1.0/(attentionWidth)*math.exp(-(pow(abs(attention_eccentricity), 2)) /
                                                  (2*pow(attentionWidth, 2))) + 0.25
    return attention

def calc_acuity(eye_eccentricity, letPerDeg):

    # Parameters from Harvey & Dumoulin (2007); 35.55556 is to make acuity at 0 degs eq. to 1
    return (1/35.555556)/(0.018*(eye_eccentricity*letPerDeg+1/0.64))

def cal_ngram_exc_input(ngram_location, ngram_weight, eye_position, attention_position, attend_width, let_per_deg, attention_skew):

    total_exc_input = 1

    # ngram activity depends on distance of ngram letters to the centre of attention and fixation, and left/right is skewed using negative/positve att_ecc
    for letter_position in ngram_location:
        attention_eccentricity = letter_position - attention_position
        eye_eccentricity = abs(letter_position - eye_position)
        attention = get_attention_skewed(attend_width, attention_eccentricity, attention_skew)
        visual_accuity = calc_acuity(eye_eccentricity, let_per_deg)
        exc_input = attention * visual_accuity
        total_exc_input = total_exc_input * exc_input

    # AL: if ngram contains more than one letter, total excitatory input is squared
    if len(ngram_location) > 1:
        total_exc_input = math.sqrt(total_exc_input)

    # AL: excitation is regulated by ngram location. Ngrams at the word edges have a higher excitatory input.
    total_exc_input = total_exc_input * ngram_weight

    return total_exc_input

def define_slot_matching_order(n_words_in_stim, fixated_position_stimulus, attend_width):

    # Slot-matching mechanism
    # MM: check len stim, then determine order in which words are matched to slots in stim
    # Words are checked in the order of its attentwght. The closer to the fixation point, the more attention weight.
    # AL: made computation dependent on position of fixated word (so we are not assuming anymore that fixation is always at the center of the stimulus)

    positions = [+1,-1,+2,-2,+3,-3] # MM: no 0 because fix position gets added elsewhere
    # AL: number of words checked depend on attention width. The narrower the attention width the fewer words matched.
    n_words_to_match = min(n_words_in_stim, (math.floor(attend_width/3)*2+1))
    # AL: add fixated position to always be checked first
    order_match_check = [fixated_position_stimulus]
    for i, p in enumerate(positions):
        if i < n_words_to_match-1:
            next_pos = fixated_position_stimulus + p
            if next_pos >= 0 and next_pos < n_words_in_stim:
                order_match_check.append(next_pos)
    #print('slots tofill:', n_words_to_match)

    return order_match_check

def sample_from_norm_distribution(mu, sigma, recog_speeding, recognized):

    if recognized:
        return int(np.round(np.random.normal(mu - recog_speeding, sigma, 1)))
    else:
        return int(np.round(np.random.normal(mu, sigma, 1)))

def find_word_edges(stimulus):
    # MM: word_edges is dict, with key is token position (from max -2 to +2, but eg. in fst fix can be 0 to +2).
    #    Entry per key is tuple w. two elements, the left & right edges, coded in letter position
    word_edges = dict()

    # AL: regex used to find indices of word edges
    p = re.compile(r'\b\w+\b', re.UNICODE)

    # Get word edges for all words starting with the word at fixation
    for i, m in enumerate(p.finditer(stimulus)):
        word_edges[i] = (m.start(),m.end()-1)

    return word_edges

def get_midword_position_for_surrounding_word(word_position, word_edges, fixated_position_in_stimulus):

    word_center_position = None
    word_position_in_stimulus = fixated_position_in_stimulus + word_position

    # AL: make sure surrounding word is included in stimulus
    if word_position_in_stimulus in word_edges.keys():
        word_slice_length = word_edges[word_position_in_stimulus][1] - word_edges[word_position_in_stimulus][0] + 1
        word_center_position = word_edges[word_position_in_stimulus][0] + round(word_slice_length/2.0) - 1

    return word_center_position

def calc_monogram_attention_sum(position_start, position_end, eye_position, attention_position, attend_width, attention_skew, let_per_deg, foveal_word):

    # this is only used to calculate where to move next when forward saccade
    sum_attention_letters = 0

    # AL: make sure letters to the left of fixated word are not included
    if foveal_word: position_start = eye_position + 1

    for letter_location in range(position_start, position_end+1):
        # print(letter_location)
        monogram_locations_weight_multiplier = 0.5
        if foveal_word:
            if letter_location == position_end:
                monogram_locations_weight_multiplier = 1. # 2.
        elif letter_location in [position_start, position_end]:
            monogram_locations_weight_multiplier = 1. # 2.

        # Monogram activity depends on distance of monogram letters to the centre of attention and fixation
        attention_eccentricity = letter_location - attention_position
        eye_eccentricity = abs(letter_location - eye_position)
        # print(attention_eccentricity, eye_eccentricity)
        attention = get_attention_skewed(attend_width, attention_eccentricity, attention_skew)
        visual_acuity = calc_acuity(eye_eccentricity, let_per_deg)
        # print(attention,visual_acuity)
        sum_attention_letters += (attention * visual_acuity) * monogram_locations_weight_multiplier

    return sum_attention_letters

def calc_word_attention_right(word_edges, eye_position, attention_position, attend_width, salience_position, attention_skew, let_per_deg, fixated_position_in_stimulus):

    # MM: calculate list of attention wgts for all words in stimulus to right of fix.
    word_attention_right = []
    attention_position += round(salience_position*attend_width)

    for i, edges in word_edges.items():

        # if n or n + x (but not n - x), so only fixated word or words to the right
        if i >= fixated_position_in_stimulus:
            # print(i, edges)
            word_start_edge = edges[0]
            word_end_edge = edges[1]

            foveal_word = False
            if i == fixated_position_in_stimulus:
                foveal_word = True

            # if eye position at last letter (right edge) of fixated word
            if foveal_word and eye_position == word_end_edge:
                # set attention wghts for (nonexisting) right part of fixated word to 0
                crt_word_monogram_attention_sum = 0
            else:
                crt_word_monogram_attention_sum = calc_monogram_attention_sum(word_start_edge, word_end_edge, eye_position, attention_position, attend_width, attention_skew, let_per_deg, foveal_word)
            # print('word position and visual salience: ',i,crt_word_monogram_attention_sum)
            word_attention_right.append(crt_word_monogram_attention_sum)
            # print(f'visual salience of {i} to the right of fixation: {crt_word_monogram_attention_sum}')
    return word_attention_right

def calc_saccade_error(saccade_distance, optimal_distance, saccErr_scaler, saccErr_sigma, saccErr_sigma_scaler,use_saccade_error):

    # TODO include fixdur, as in EZ and McConkie (smaller sacc error after longer fixations)
    saccade_error = (optimal_distance - abs(saccade_distance)) * saccErr_scaler
    saccade_error_sigma = saccErr_sigma + (abs(saccade_distance) * saccErr_sigma_scaler)
    saccade_error_norm = np.random.normal(saccade_error, saccade_error_sigma, 1)
    if use_saccade_error:
        return saccade_error_norm
    else:
        return 0.

def check_previous_refixations_at_position(all_data, fixation, fixation_counter, max_n_refix):

    # AL: mechanism to prevent infinite refixations in words that do not get sufficient activation to get recognized
    # or are long and have higher atetntional input than surrounding words
    refixate = False
    # AL: if first fixation on text, no previous fixation to check, so refixation is allowed
    if fixation_counter == 0: refixate = True
    else:
        for i in range(1,max_n_refix+1):
            if fixation_counter - i in all_data.keys():
                if all_data[fixation_counter - i]['saccade_type'] != 'refixation' and all_data[fixation_counter - i]['foveal_word_index'] == fixation:
                    refixate = True
                    break
    return refixate