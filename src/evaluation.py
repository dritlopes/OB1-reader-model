import pandas as pd
import numpy as np
from collections import defaultdict
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sb
from utils import get_pred_dict, get_word_freq, pre_process_string
import math

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

def convert_trial_to_word_level(stimuli):

    stimuli_words = defaultdict(list)
    for i, text_info in stimuli.iterrows():
        word_ids = text_info['word_ids'].replace('[','').replace(']','').split(',')
        words = text_info['words'].replace('[','').replace(']','').split(',')
        for word_info in zip(word_ids, words):
            stimuli_words['Text_ID'].append(int(text_info['id']))
            stimuli_words['Word_Number'].append(word_info[0])
            stimuli_words['Word'].append(word_info[1].replace("'",""))
    return pd.DataFrame(stimuli_words)

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

    # single fixation average duration
    single_fixation_duration_sim = single_fixation_sim.groupby(['text_id', 'word_id'])[
        ['fixation duration']].mean().reset_index()
    single_fix_duration = single_fixation_duration_sim.rename(columns={'fixation duration': 'single_fix_duration'})

    return single_fix_duration

def compute_first_fixation(first_pass:pd.DataFrame):

    first_fixation = first_pass.loc[first_pass.groupby(['simulation_id', 'text_id', 'word_id']).apply(lambda x: x.index[0]).values, :]
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

def get_word_factors(pm, eye_movements_df):

    # predictability
    pred_map = get_pred_dict(pm,None)
    pred_column = []
    for i, item in eye_movements_df.iterrows():
        pred_value = 0.0
        if str(item['text_id']) in pred_map.keys():
            if str(item['word_id']) in pred_map[str(item['text_id'])].keys():
                word = item['word'].strip()
                if word in pred_map[str(item['text_id'])][str(item['word_id'])].keys():
                    pred_value = pred_map[str(item['text_id'])][str(item['word_id'])][word]
        pred_column.append(pred_value)

    # frequency
    freq_map = get_word_freq(pm,None)
    freq_column = []
    for i, item in eye_movements_df.iterrows():
        freq_value = 0.0
        word = item['word'].strip()
        if word in freq_map.keys():
            freq_value = freq_map[word]
        freq_column.append(freq_value)

    # length
    length_column = []
    for i, item in eye_movements_df.iterrows():
        word = item['word'].strip()
        length_column.append(len(word))

    # merge all
    eye_movements_df['predictability'] = pred_column
    eye_movements_df['frequency'] = freq_column
    eye_movements_df['length'] = length_column

    return eye_movements_df

def drop_nan_values(true_values:pd.Series, simulated_values:pd.Series):

    values = defaultdict(list)
    counter = 0
    for true_value, simulated_value in zip(true_values, simulated_values):
        if not true_values.isnull()[counter] and not simulated_values.isnull()[counter]:
            values['true'].append(true_value)
            values['simulated'].append(simulated_value)
        counter += 1
    return values

# ---------------- Score functions ------------------
def compute_root_mean_squared_error(true_values:list, simulated_values:list):

    # mean2error = math.sqrt(sum([(simulated_value - true_value) ** 2 for simulated_value, true_value in
    #                   zip(simulated_values, true_values)]) / len(simulated_values))
    root_mean2error = math.sqrt(np.square(np.subtract(simulated_values, true_values)).mean())
    return root_mean2error

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

        # read in eye_tracking data for comparing simulation with observed measures
        # encoding = chardet.detect(open(parameters.eye_tracking_filepath, "rb").read())['encoding']
        eye_tracking = pd.read_csv(parameters.eye_tracking_filepath, encoding="ISO-8859-1")
        # remove rows where Word_Number is nan value, to be able to convert floats to ints
        eye_tracking = eye_tracking.dropna(subset=['Word_Number'])
        # - 1 to be compatible with index used in simulation (starting from 0, instead of 1)
        eye_tracking['Word_Number'] = eye_tracking['Word_Number'].astype(int).apply(lambda x: x - 1)
        eye_tracking['Text_ID'] = eye_tracking['Text_ID'].apply(lambda x: x - 1)
        eye_tracking['Word'] = [pre_process_string(word) for word in eye_tracking['Word']]
        # only including first two texts for now
        eye_tracking = eye_tracking[eye_tracking['Text_ID'] < 2]

        # get word-level eye-movement measures, averaged over simulations
        stimuli = eye_tracking[['Text_ID', 'Word_Number', 'Word']]
        # TODO add parameter keep_participant_level
        eye_movement_measures = compute_word_level_eye_movements(simulation_output, stimuli, parameters.evaluation_measures)
        # get word factors: frequency, length, predictability
        eye_movement_measures = get_word_factors(parameters, eye_movement_measures)
        filepath = output_filepath.replace('simulation_', 'simulation_eye_movements_')
        eye_movement_measures.to_csv(filepath, sep='\t', index_label='id')

        # get word level measures from eye_tracking, averaged over participants
        observed_eye_movement_measures = compute_observed_word_level_eye_movements(eye_tracking, parameters.evaluation_measures)
        observed_eye_movement_measures = get_word_factors(parameters, observed_eye_movement_measures)
        filepath = output_filepath.replace('simulation_', 'observed_eye_movements_')
        observed_eye_movement_measures.to_csv(filepath, sep='\t', index_label='id')

        # mean square error between each measure in simulation and eye-tracking
        mean2errors = defaultdict(list)
        for measure in parameters.evaluation_measures:
            assert len(observed_eye_movement_measures[measure]) == len(eye_movement_measures[measure])
            # excluding words with nan value, e.g. skipped words of prob 1. Should words that are always skipped be included in the equation?
            values = drop_nan_values(observed_eye_movement_measures[measure], eye_movement_measures[measure])
            mean2error = compute_root_mean_squared_error(values['true'], values['simulated'])
            mean2errors['eye_tracking_measure'].append(measure)
            mean2errors['mean_squared_error'].append(mean2error)
        filepath = output_filepath.replace('simulation_', 'root_mean2error_eye_movements_')
        mean2error_df = pd.DataFrame(mean2errors)
        mean2error_df.to_csv(filepath, sep='\t', index=False)

        # plot predictability vs. measure
        predictor_sim = ['OB1-reader' for i in range(len(eye_movement_measures))]
        eye_movement_measures["predictor"] = predictor_sim
        predictor_obs = ['PROVO' for i in range(len(observed_eye_movement_measures))]
        observed_eye_movement_measures["predictor"] = predictor_obs
        # merge model + eye-tracking
        data = pd.concat([eye_movement_measures, observed_eye_movement_measures], axis=0).reset_index()
        for measure in parameters.evaluation_measures:
            plot = sb.relplot(data=data, x=measure, y='predictability', hue='predictor', kind='line')
            plot.figure.savefig(f"../results/plot_{measure}_{parameters.prediction_flag}.png")

        # TODO get word-level measures with participant-level preserved for stat tests

        # TODO fit predictability effects
        # for measure in parameters.evaluation_measures:
        #     groups = [eye_movement_measures['participant_id'],eye_movement_measures['text_id'],eye_movement_measures['word_id']]
        #     stats_model = smf.mixedlm(f"{measure} ~ predictability", eye_movement_measures, groups=groups)
        #     stats_result = stats_model.fit(method=["lbfgs"])



