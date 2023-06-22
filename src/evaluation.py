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

# ---------------- Simulation eye-movement measures ------------------
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
    # first_pass.to_csv('../results/first_pass.csv')

    return first_pass

def get_text_words(stimuli):

    words_in_text = defaultdict(list)
    for word_info, rows in stimuli.groupby(['Text_ID', 'Word_Number']):
        words_in_text[int(word_info[0])].append(rows['Word'].tolist()[0])
    return words_in_text

def get_skipped_words(first_pass:pd.DataFrame, words_in_text:defaultdict):

    """
    For each simulation, compare token ids in first pass with token ids from specific text. The token indicices not in first pass are considered skipped
    :param first_pass: pandas dataframe containing fixations in first-pass
    :param words_in_text: list of words in text
    :return: pandas dataframe with only words skipped by in at least one simulation
    """
    skipped_words = defaultdict(list)

    for sim_id, sim_hist in first_pass.groupby('simulation_id'):
        for text_id, text_hist in sim_hist.groupby('text_id'):
            for word_id, word in enumerate(words_in_text[text_id], 1):
                if word_id not in list(text_hist['word_id'].unique()):
                    skipped_words['simulation_id'].append(sim_id)
                    skipped_words['text_id'].append(text_id)
                    skipped_words['word_id'].append(word_id)
                    skipped_words['word'].append(word)

    skipped_words = pd.DataFrame.from_dict(skipped_words)

    return skipped_words

def compute_skipping_probability(first_pass:pd.DataFrame, skipped_words:pd.DataFrame):

    """
    Given the first-pass fixations and the words skipped by at least one simulation, compute the proportion of
    simulations in which each word was skipped.
    :param first_pass: pandas dataframe containing fixations in first-pass
    :param skipped_words: pandas data frame containing the words skipped in first-pass
    :return: pandas dataframe with the skipping proportion per word
    """

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

def compute_single_fix_probability(first_pass:pd.DataFrame):

    """
    Get single fixation probabilities for each word.
    :param first_pass: pandas dataframe containing fixations in first-pass
    :return: pandas dataframe with single fixation probabilitiies per word
    """

    # filter each word's first-pass fixations by those of length 1, thus fixated only once
    single_fixation_sim = first_pass.groupby(['simulation_id', 'text_id', 'word_id']).filter(lambda x: len(x) == 1).reset_index()
    single_fixation_sim = single_fixation_sim[['simulation_id', 'text_id', 'word', 'word_id', 'fixation duration']]
    # single_fixation_sim.to_csv('../results/single_fix.csv')

    # single fixation probability
    single_fix_dict = defaultdict(list)
    #single_fix = [w for w, h in single_fixation_sim.groupby(['text_id', 'word_id'])]

    for unique_word, hist in first_pass.groupby(['text_id', 'word_id']):
        prob = 0.0
        single_fix = single_fixation_sim.query(f"text_id=={unique_word[0]} & word_id=={unique_word[1]}")['simulation_id']
        if len(single_fix) > 0:
            #single_hist = single_fix['simulation_id']
            prob = len(single_fix) / len(hist['simulation_id'].unique())
        single_fix_dict['text_id'].append(unique_word[0])
        single_fix_dict['word_id'].append(unique_word[1])
        single_fix_dict['word'].append(hist['word'].tolist()[0])
        single_fix_dict['single_fix_proportion'].append(prob)
    single_fix_df = pd.DataFrame.from_dict(single_fix_dict)

    # check no word is repeated, i.e. each word (text_id+word_id) should appear once with one probability value
    assert single_fix_df.groupby(['text_id', 'word_id']).value_counts().unique().tolist() == [1]

    return single_fix_df

def compute_single_fix_duration(first_pass:pd.DataFrame):

    """
    Get single fixation duration for each word.
    :param first_pass: pandas dataframe containing fixations in first-pass
    :return: pandas dataframe with single fixation durations per word
    """

    # filter each word's first-pass fixations by those of length 1, thus fixated only once
    single_fixation_sim = first_pass.groupby(['simulation_id', 'text_id', 'word_id']).filter(
        lambda x: len(x) == 1).reset_index()
    single_fixation_sim = single_fixation_sim[['simulation_id', 'text_id', 'word', 'word_id', 'fixation duration']]

    # single fixation average duration
    single_fixation_duration_sim = single_fixation_sim.groupby(['text_id', 'word_id'])[
        ['fixation duration']].mean().reset_index()
    single_fix_duration = single_fixation_duration_sim.rename(columns={'fixation duration': 'single_fix_duration'})

    return single_fix_duration

def compute_first_fixation(first_pass:pd.DataFrame):

    first_fixation = first_pass.loc[first_pass.groupby(['simulation_id', 'text_id', 'word_id']).apply(lambda x: x.index[0]).values, :]
    first_fixation = first_fixation[['simulation_id', 'text_id', 'word', 'word_id', 'fixation duration']]
    first_fixation_duration_sim = first_fixation.groupby(['text_id', 'word_id'])[['fixation duration']].mean().reset_index()
    first_fix_duration = first_fixation_duration_sim.rename(columns={'fixation duration': 'first_fix_duration'})

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

def compute_word_level_eye_movements(simulation_output:pd.DataFrame, stimuli:pd.DataFrame, measures:list):

    """
    Transform fixation data into word-centred data, where each row is a word in each text/trial.
    Add columns with word-level eye movement measures.
    :param simulation_output: contains the fixations of the model simulation(s)
    :return: a word-centred pandas data frame with word-level eye movement measures
    """
    first_pass = get_first_pass_fixations(simulation_output)
    result_dfs = dict()
    for measure in measures:
        if measure == 'skipping_proportion':
            text_words = get_text_words(stimuli)
            skipped_words = get_skipped_words(first_pass, text_words)
            to_add = compute_skipping_probability(first_pass, skipped_words)
        elif measure == 'single_fix_proportion':
            to_add = compute_single_fix_probability(first_pass)
        elif measure == 'single_fix_duration':
            to_add = compute_single_fix_duration(first_pass)
        elif measure == 'first_fix_duration':
            to_add = compute_first_fixation(first_pass)
        elif measure == 'gaze_duration':
            to_add = compute_gaze_duration(first_pass)
        elif measure == 'total_reading_time':
            to_add = compute_total_reading_time(simulation_output)
        result_dfs[measure] = to_add

    # merge all, such that each column is one eye-movement measure and each row is a word
    result_columns = defaultdict(list)
    for item, hist in stimuli.groupby(['Text_ID', 'Word_Number']):
        if item[0] in simulation_output["text_id"].tolist():
            result_columns['text_id'].append(item[0])
            result_columns['word_id'].append(item[1])
            result_columns['word'].append(hist['Word'].tolist()[0])
            for measure in measures:
                cell = result_dfs[measure].query(f"text_id=={item[0]} & word_id=={item[1]}")[measure]
                if len(cell) > 0:
                    result_columns[measure].append(cell.item())
                else:
                    result_columns[measure].append(np.nan)
    results = pd.DataFrame(result_columns)

    return results

# ---------------- Observed eye-movement measures ------------------
def compute_obs_skipping_probability(eye_tracking: pd.DataFrame):

    skipping = eye_tracking.groupby(['Text_ID', 'Word_Number'])['IA_SKIP'].value_counts(
        normalize=True).reset_index(name='skipping_proportion')
    skipping = skipping[skipping['IA_SKIP'] == 1]
    skipping = skipping.drop(labels='IA_SKIP', axis=1)
    skipping = skipping.rename(columns={'Text_ID': 'text_id', 'Word_Number': 'word_id'})

    return skipping

def compute_obs_single_fix_probability(eye_tracking: pd.DataFrame):

    single_fix = []
    # determine which words were fixated only once by each participant
    for word, hist in eye_tracking.groupby(['Participant_ID', 'Text_ID', 'Word_Number']):
        # single fix is defined as: if gaze duration equals first fixation duration,
        # then word was fixated only once in first pass
        if hist['IA_FIRST_RUN_DWELL_TIME'].tolist()[0] == hist['IA_FIRST_FIXATION_DURATION'].tolist()[0]:
            single_fix.append(1)
        else:
            single_fix.append(0)
    # add binary single fix column
    eye_tracking['IA_SINGLE_FIX'] = single_fix
    # count the
    single_fix_probs_provo = eye_tracking.groupby(['Text_ID', 'Word_Number'])['IA_SINGLE_FIX'].value_counts(
        normalize=True).reset_index(name='single_fix_proportion')
    # only keep the proportion of single fix (excluding proportion of not single fix)
    single_fix_probs_provo = single_fix_probs_provo[single_fix_probs_provo['IA_SINGLE_FIX'] == 1]
    # drop binary single fix column, we only want the proportion
    single_fix_proportion_obs = single_fix_probs_provo.drop(labels='IA_SINGLE_FIX', axis=1)
    single_fix_proportion_obs = single_fix_proportion_obs.rename(columns={'Text_ID': 'text_id', 'Word_Number': 'word_id'})

    return single_fix_proportion_obs

def compute_obs_single_fix_duration(eye_tracking: pd.DataFrame):

    eye_tracking_single_fix = eye_tracking[eye_tracking['IA_SINGLE_FIX'] == 1]
    eye_tracking_single_fix = eye_tracking_single_fix.groupby(['Text_ID', 'Word_Number'])[
        ['IA_FIRST_FIXATION_DURATION']].mean().reset_index()
    eye_tracking_single_fix = eye_tracking_single_fix.rename(
        columns={'IA_FIRST_FIXATION_DURATION': 'single_fix_duration', 'Text_ID': 'text_id', 'Word_Number': 'word_id'})

    return eye_tracking_single_fix

def compute_obs_first_fix_duration(eye_tracking: pd.DataFrame):

    eye_tracking_fix_duration = eye_tracking.groupby(['Text_ID', 'Word_Number'])['IA_FIRST_FIXATION_DURATION'].mean().reset_index()
    eye_tracking_fix_duration = eye_tracking_fix_duration.rename(columns={'IA_FIRST_FIXATION_DURATION': 'first_fix_duration',
                                    'Text_ID': 'text_id', 'Word_Number': 'word_id'})
    return eye_tracking_fix_duration

def compute_obs_gaze_duration(eye_tracking: pd.DataFrame):

    gaze_duration_obs = eye_tracking.groupby(['Text_ID', 'Word_Number'])[
        ['IA_FIRST_RUN_DWELL_TIME']].mean().reset_index()
    gaze_duration_obs = gaze_duration_obs.rename(columns={'IA_FIRST_RUN_DWELL_TIME': 'gaze_duration',
                                                          'Text_ID': 'text_id', 'Word_Number': 'word_id'})

    return gaze_duration_obs

def compute_obs_total_reading_time(eye_tracking:pd.DataFrame):
    
    total_fixation_duration_obs = eye_tracking.groupby(['Text_ID', 'Word_Number'])[
        ['IA_DWELL_TIME']].mean().reset_index()
    total_fixation_duration_obs = total_fixation_duration_obs.rename(columns={'IA_DWELL_TIME': 'total_reading_time',
                                                                              'Text_ID': 'text_id', 'Word_Number': 'word_id'})
    
    return total_fixation_duration_obs

def compute_observed_word_level_eye_movements(eye_tracking: pd.DataFrame, measures:list):

    result_dfs = dict()

    for measure in measures:

        if measure == 'skipping_proportion':
            to_add = compute_obs_skipping_probability(eye_tracking)
        elif measure == 'single_fix_proportion':
            to_add = compute_obs_single_fix_probability(eye_tracking)
        elif measure == 'single_fix_duration':
            to_add = compute_obs_single_fix_duration(eye_tracking)
        elif measure == 'first_fix_duration':
            to_add = compute_obs_first_fix_duration(eye_tracking)
        elif measure == 'gaze_duration':
            to_add = compute_obs_gaze_duration(eye_tracking)
        elif measure == 'total_reading_time':
            to_add = compute_obs_total_reading_time(eye_tracking)

        result_dfs[measure] = to_add

    result_columns = defaultdict(list)
    # merge all
    for item, hist in eye_tracking.groupby(['Text_ID', 'Word_Number']):
        result_columns['text_id'].append(item[0])
        result_columns['word_id'].append(item[1])
        result_columns['word'].append(hist['Word'].tolist()[0])
        for measure in measures:
            cell = result_dfs[measure].query(f"text_id=={item[0]} & word_id=={item[1]}")[measure]
            if len(cell) > 0:
                result_columns[measure].append(cell.item())
            else:
                result_columns[measure].append(np.nan)
    results = pd.DataFrame(result_columns)

    return results

# ---------------- Score functions ------------------
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

# ---------------- MAIN ------------------
def evaluate_output (parameters, output_filepath:str):

    simulation_output = pd.read_csv(output_filepath, sep='\t')
    simulation_output = simulation_output.rename(columns={'foveal word index': 'word_id',
                                                          'foveal word': 'word'})
    if parameters.task_to_run == 'continuous reading':

        # exclude first word of every passage (not in eye tracking -PROVO- data either)
        simulation_output = simulation_output[simulation_output['word_id'] != 0]

        # read in stimulus to add skipped words
        # encoding = chardet.detect(open(parameters.stimuli_filepath, "rb").read())['encoding']
        stimuli = pd.read_csv(parameters.stimuli_filepath, encoding="ISO-8859-1")
        # - 1 to be compatible with index used in simulation (starting from 0, instead of 1)
        stimuli['Word_Number'] = stimuli['Word_Number'].astype(int).apply(lambda x: x - 1)
        stimuli['Text_ID'] = stimuli['Text_ID'].apply(lambda x: x - 1)

        # get word-level eye-movement measures, averaged over simulations
        eye_movement_measures = compute_word_level_eye_movements(simulation_output, stimuli, parameters.evaluation_measures)
        filepath = output_filepath.replace('simulation_', 'simulation_eye_movements_')
        eye_movement_measures.to_csv(filepath, sep='\t', index=False)

        # read in eye_tracking data for comparing simulation with observed measures
        # encoding = chardet.detect(open(parameters.eye_tracking_filepath, "rb").read())['encoding']
        eye_tracking = pd.read_csv(parameters.eye_tracking_filepath, encoding="ISO-8859-1")
        # remove rows where Word_Number is nan value, to be able to convert floats to ints
        eye_tracking = eye_tracking.dropna(subset=['Word_Number'])
        # - 1 to be compatible with index used in simulation (starting from 0, instead of 1)
        eye_tracking['Word_Number'] = eye_tracking['Word_Number'].astype(int).apply(lambda x: x - 1)
        eye_tracking['Text_ID'] = eye_tracking['Text_ID'].apply(lambda x: x - 1)
        # only including first two texts for now
        eye_tracking = eye_tracking[eye_tracking['Text_ID'] < 2]

        # get word level measures from eye_tracking, averaged over participants
        observed_eye_movement_measures = compute_observed_word_level_eye_movements(eye_tracking, parameters.evaluation_measures)
        filepath = output_filepath.replace('simulation_', 'observed_eye_movements_')
        observed_eye_movement_measures.to_csv(filepath, sep='\t', index=False)
