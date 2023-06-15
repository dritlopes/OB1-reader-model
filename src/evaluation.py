import numpy as np
import pandas as pd
import pickle
import json
import seaborn as sb
import numpy as np
from collections import defaultdict
import sys
import chardet
import matplotlib.pyplot as plt


def get_first_pass_fixations(simulation_df:pd.DataFrame):

    """
    Gather first-pass fixation to determine word-measures that are based on first-pass fixations, e.g. skipping, first fixation, single fixation and gaze duration.
    First-pass fixations of a word are defined as the first fixation on that word resulted from a forward or wordskip saccade, plus all sequential refixations.
    :param simulation_output: contains the fixations of the model simulation(s)
    :return: a pandas dataframe with only the fixations belonging to first-pass
    """

    first_pass_indices = []

    for sim_id, sim_hist in simulation_df.groupby('simulation_id'):
        for text_id, text_hist in sim_hist.groupby('text_id'):
            for word, fix_hist in text_hist.groupby('word_id'):
                # make sure the eyes have NOT moved to beyond the word before fixating it for the first time
                if all(word > previous_word for previous_word in
                       text_hist['word_id'].tolist()[:fix_hist['fixation_counter'].tolist()[0] - 1]):
                    # check each fixation on the word
                    for i, fix in fix_hist.iterrows():
                        # if fixation was not the result of a regression
                        if fix['saccade type'] != 'regression':
                            # add to first-pass if first fixation on word
                            if fix['fixation_counter'] == fix_hist['fixation_counter'].tolist()[0]:
                                first_pass_indices.append(i)
                            # add to first-pass the following fixations which are sequential to first fixation
                            elif i == first_pass_indices[-1] + 1:
                                first_pass_indices.append(i)

    first_pass = simulation_df.filter(items=first_pass_indices, axis=0)

    return first_pass

def get_skipped_words(first_pass:pd.DataFrame, words_in_text):

    """
    For each simulation, compare token ids in first pass with token ids from specific text. The token indicices not in first pass are considered skipped
    :param first_pass: pandas dataframe containing fixations in first-pass
    :param words_in_text:
    :return: pandas dataframe with only words skipped by in least one simulation
    """
    skipped_words = defaultdict(list)

    for sim_id, sim_hist in first_pass.groupby('simulation_id'):
        for text_id, text_hist in sim_hist.groupby('text_id'):
            for word_id, word in enumerate(words_in_text[text_id + 1], 1):
                if word_id not in list(text_hist['word_id'].unique()):
                    skipped_words['simulation_id'].append(sim_id)
                    skipped_words['text_id'].append(text_id)
                    skipped_words['word_id'].append(word_id)
                    skipped_words['word'].append(word)

    skipped_words = pd.DataFrame.from_dict(skipped_words)

    return skipped_words

def compute_skipping_probability(first_pass:pd.DataFrame, skipped_words:pd.DataFrame):

    skipping_probs = defaultdict(list)

    for unique_word, hist in first_pass.groupby(['text_id', 'word_id']):
        # word has skipping probability 0 if it was fixated in first pass by all simulations/participants
        if len(hist['simulation_id'].unique()) == len(first_pass['simulation_id'].unique()):
            prob = 0.0
            skipping_probs['text_id'].append(unique_word[0])
            skipping_probs['word_id'].append(unique_word[1])
            skipping_probs['word'].append(hist['word'].tolist()[0])
            skipping_probs['skipping_proportion'].append(prob)

    # words in skipped_words are the ones that do not appear in first pass by at least one simulation/participant
    for unique_word, hist in skipped_words.groupby(['text_id', 'word_id']):
        # probability is the number of times word was skipped over number of simulations/participants
        prob = len(hist) / len(skipped_words['simulation_id'].unique())
        skipping_probs['text_id'].append(unique_word[0])
        skipping_probs['word_id'].append(unique_word[1])
        skipping_probs['word'].append(hist['word'].tolist()[0])
        skipping_probs['skipping_proportion'].append(prob)

    skipping_probs = pd.DataFrame.from_dict(skipping_probs)
    # check no word is repeated, i.e. each word (text_id+word_id) should appear once with one probability value
    assert skipping_probs.groupby(['text_id', 'word_id']).value_counts().unique().tolist() == [1]

    skipping_sim = skipping_probs.sort_values(by=['text_id', 'word_id']).reset_index(drop=True)

    return skipping_sim

def compute_single_fix(first_pass:pd.DataFrame):

    """
    Get single fixation probabilities and average duration for each word.
    :param first_pass: pandas dataframe containing fixations in first-pass
    :return: pandas dataframe with single fixation probabiltiies and average duration per word
    """

    # filter each word's first-pass fixations by those of length 1, thus fixated only once
    single_fixation_sim = first_pass.groupby(['simulation_id', 'text_id', 'word_id']).filter(lambda x: len(x) == 1).reset_index()
    single_fixation_sim = single_fixation_sim[['simulation_id', 'text_id', 'word', 'word_id', 'fixation duration']]

    # single fixation average duration
    single_fixation_duration_sim = single_fixation_sim.groupby(['text_id', 'word_id'])[['fixation duration']].mean().reset_index()
    single_fix_duration = single_fixation_duration_sim.rename(columns={'fixation duration': 'single_fix_duration'})

    # single fixation probability
    single_fix_probs = defaultdict(list)
    single_fix = [w for w, h in single_fixation_sim.groupby(['text_id', 'word_id'])]

    for unique_word, hist in first_pass.groupby(['text_id', 'word_id']):
        prob = 0.0
        if tuple(unique_word) in single_fix:
            prob = len(hist) / len(hist['simulation_id'].unique())
        single_fix_probs['text_id'].append(unique_word[0])
        single_fix_probs['word_id'].append(unique_word[1])
        single_fix_probs['word'].append(hist['word'].tolist()[0])
        single_fix_probs['singe_fix_proportion'].append(prob)

    single_fix_proportion = pd.DataFrame.from_dict(single_fix_probs)

    # check no word is repeated, i.e. each word (text_id+word_id) should appear once with one probability value
    assert single_fix_proportion.groupby(['text_id', 'word_id']).value_counts().unique().tolist() == [1]

    single_fix_df = pd.concat([single_fix_proportion,single_fix_duration], axis=1)

    return single_fix_df

def compute_first_fixation(first_pass:pd.DataFrame):

    first_fixation = first_pass.loc[first_pass.groupby(['simulation_id', 'text_id', 'word_id']).apply(lambda x: x.index[0]).values, :]
    first_fixation = first_fixation[['simulation_id', 'text_id', 'word', 'word_id', 'fixation duration']]
    first_fixation_duration_sim = first_fixation.groupby(['text_id', 'word_id'])[['fixation duration']].mean().reset_index()
    first_fix_duration = first_fixation_duration_sim.rename(columns={'fixation duration': 'first_fixation_duration'})

    return first_fix_duration

def compute_gaze_duration(first_pass:pd.DataFrame):

    gaze_duration_sim = first_pass.groupby(['simulation_id', 'text_id', 'word_id'])[['fixation duration']].sum().reset_index()
    gaze_duration_sim = gaze_duration_sim.groupby(['text_id', 'word_id'])[['fixation duration']].mean().reset_index()
    gaze_duration = gaze_duration_sim.rename(columns={'fixation duration': 'gaze_duration'})

    return gaze_duration

def compute_total_reading_time(simulation_output:pd.DataFrame):

    sum_fixation_duration_sim = simulation_output.groupby(['simulation_id', 'text_id', 'word_id'])[['fixation duration']].sum().reset_index()
    total_fixation_duration_sim = sum_fixation_duration_sim.groupby(['text_id', 'word_id'])[['fixation duration']].mean().reset_index()
    total_reading_duration = total_fixation_duration_sim.rename(columns={'fixation duration': 'total_reading_time'})

    return total_reading_duration

def compute_mean_squared_error(true_values:list, simulated_values:list):
    mean2error = sum([(simulated_value - true_value) ** 2 for simulated_value, true_value in
                      zip(simulated_values, true_values)]) / len(simulated_values)
    return mean2error

def compute_mean_squared_error_of_word(true_value:float, simulated_value:float, standard_deviation:float):
    mean2error = ((simulated_value - true_value) / standard_deviation) ** 2
    return mean2error

def normalize(sample:list):
    normalized = (sample - min(sample)) / (max(sample) - min(sample))
    return normalized

def compute_word_level_eye_movement_measures(simulation_output:pd.DataFrame):

    """
    Transform fixation data into word-centred data, where each row is a word in each text/trial.
    Add columns with word-level eye movement measures.
    :param simulation_output: contains the fixations of the model simulation(s)
    :return: a word-centred pandas data frame with word-level eye movement measures
    """

    measures = ['skipping_proportion',
                'single_fix_proportion',
                'single_fix_duration',
                'first_fix_duration',
                'gaze_duration',
                'total_reading_time']
    first_pass = get_first_pass_fixations(simulation_output)
    results = pd.DataFrame()

    for measure in measures:

        if measure == 'skipping_proportion':
            skipped_words = get_skipped_words(first_pass, words_in_text=None) # TODO get words in text
            to_add = compute_skipping_probability(first_pass,skipped_words)
        elif measure == 'single_fix_proportion':
            to_add = compute_single_fix(first_pass)
        elif measure == 'first_fix_duration':
            to_add = compute_first_fixation(first_pass)
        elif measure == 'gaze_duration':
            to_add = compute_gaze_duration(first_pass)
        elif measure == 'total_reading_time':
            to_add = compute_total_reading_time(simulation_output)

        results = pd.concat([results,to_add],axis=1)
        results.drop_duplicates()
        # TODO check alignment of words, add word column to dataframes
    return results

def evaluate_output (parameters, output_filepath:str):

    simulation_output = pd.read_csv(output_filepath, sep='\t')
    # TODO rename columns in simulation df
    if parameters.task == 'continuous reading':
        # exclude first word of every passage (not in eye tracking -PROVO- data either)
        simulation_output = simulation_output[simulation_output['foveal word index'] != 0]
        # get word-level eye-movement measures
        eye_movement_measures = compute_word_level_eye_movement_measures(simulation_output)