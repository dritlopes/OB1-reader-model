import pandas as pd
import numpy as np
from collections import defaultdict
import statsmodels.api as sm
import scipy.stats as stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sb
import ptitprince as pt
from utils import get_pred_dict, get_word_freq, pre_process_string
import math
import os
import warnings
from collections import Counter
import itertools

# ---------------- Simulation eye-movement measures ------------------
def pre_process_eye_tracking(eye_tracking: pd.DataFrame, eye_tracking_filepath: str, stimuli):

    if 'provo' in eye_tracking_filepath.lower():

        # make sure all columns of interest are named as code expects
        eye_tracking = eye_tracking.rename(columns={'Word_Unique_ID': 'id',
                                                     'Text_ID': 'text_id',
                                                     'Word_Number': 'word_id',
                                                     'Participant_ID': 'participant_id',
                                                     'Word': 'word',
                                                     'IA_SKIP': 'skip',
                                                     'IA_FIRST_FIXATION_DURATION': 'first_fix_duration',
                                                     'IA_FIRST_RUN_DWELL_TIME': 'gaze_duration',
                                                     'IA_DWELL_TIME': 'total_reading_time',
                                                     'IA_REGRESSION_IN': 'regression_in',
                                                     'IA_REGRESSION_OUT': 'regression_out'})

        # remove rows where Word_Number is nan value, to be able to convert floats to ints
        eye_tracking = eye_tracking.dropna(subset=['word_id'])

        # - 1 to be compatible with index used in simulation (starting from 0, instead of 1)
        eye_tracking['word_id'] = eye_tracking['word_id'].astype(int).apply(lambda x: x - 1)
        eye_tracking['text_id'] = eye_tracking['text_id'].apply(lambda x: x - 1)

        # fix some errors in provo eye_tracking data
        eye_tracking.loc[(eye_tracking['word_id'] == 2) & (eye_tracking['word'] == 'evolution') & (eye_tracking['text_id'] == 17), 'word_id'] = 50
        for row in [{'text': 2, 'start': 45, 'end': 59},
                    {'text': 12, 'start': 19, 'end': 54}]:
            for i in range(row['start'], row['end']+1):
                eye_tracking.loc[(eye_tracking['word_id'] == i) & (eye_tracking['text_id'] == row['text']), 'word_id'] = i - 1

        # check alignment between model stimuli and eye-tracking words
        for text, fixations in eye_tracking.groupby('text_id'):
            stimuli_text = stimuli.iloc[int(text)]
            text_words = stimuli_text['words'].replace('[', '').replace(']', '').replace(',', '').replace("'", "").split()
            text_word_ids = str(stimuli_text['word_ids']).replace('[', '').replace(']', '').replace(',', '').replace("'", "").split()
            for participant, hist in fixations.groupby('participant_id'):
                eye_tracking_word_dict = dict()
                for et_word, et_word_id in zip(hist['word'].tolist(), hist['word_id'].tolist()):
                    eye_tracking_word_dict[et_word_id] = et_word
                for word, word_id in zip(text_words[1:], text_word_ids[1:]):
                    # a few exceptions we know are completely missing in eye-tracking
                    if (int(text), word, int(word_id)) not in [(54, 'a', 9), (54, 'profession', 60), (54, 'writing', 61)]:
                        if int(word_id) in eye_tracking_word_dict.keys():
                            eye_tracking_word = eye_tracking_word_dict[int(word_id)]
                            if word != pre_process_string(eye_tracking_word) and word not in ['90','doesnÃµt','womenÃµs','bondsÃµ']:
                                warnings.warn(
                                    f'Word in eye tracking "{eye_tracking_word}" from participant {participant} not the same as word in model stimuli "{word}" in text {text}, position {word_id}')
                        else:
                            warnings.warn(f'Position {word_id}, "{word}", in text {text} not found in eye tracking data from participant {participant}')

        # only consider first fix and gaze duration from firs-pass fixations
        first_fix, gaze_duration = [], []
        for word, hist in eye_tracking.groupby(['participant_id', 'text_id', 'word_id']):
            # if word was skipped at first-pass, then fixations are not at first-pass
            if hist['skip'].tolist()[0] == 1:
                first_fix.append(None)
                gaze_duration.append(None)
            else:
                first_fix.append(hist['first_fix_duration'].tolist()[0])
                gaze_duration.append(hist['gaze_duration'].tolist()[0])
        eye_tracking['first_fix_duration'] = first_fix
        eye_tracking['gaze_duration'] = gaze_duration

        # add single fix
        single_fix, single_fix_dur = [], []
        # determine which words were fixated only once by each participant
        for word, hist in eye_tracking.groupby(['participant_id', 'text_id', 'word_id']):
            # single fix is defined as: if gaze duration equals first fixation duration,
            # then word was fixated only once in first pass
            assert len(hist) == 1, print(f'Word id {word} appears more than once in eye-tracking data {hist}')
            # if word has been fixated at first pass (not nan value)
            if not np.isnan(hist['gaze_duration'].tolist()[0]) and not np.isnan(hist['first_fix_duration'].tolist()[0]):
                # if first fix and gaze duration are the same, word has been singly fixated in first pass
                if hist['gaze_duration'].tolist()[0] == hist['first_fix_duration'].tolist()[0]:
                    single_fix.append(1)
                    single_fix_dur.append(hist['first_fix_duration'].tolist()[0])
                # if not, word has been refixated in first pass
                else:
                    single_fix.append(0)
                    single_fix_dur.append(None)
            # if not fixated at first pass, single fix is None
            else:
                single_fix.append(None)
                single_fix_dur.append(None)
        # add binary single fix column
        eye_tracking['single_fix'] = single_fix
        eye_tracking['single_fix_duration'] = single_fix_dur

        # convert total reading time = 0 to NaN so it does not get included in the averages
        eye_tracking['total_reading_time'] = eye_tracking['total_reading_time'].apply(lambda x: None if x==0 else x)

        # add item level id (word id per participant)
        item_id = []
        for participant, hist in eye_tracking.groupby('participant_id'):
            item_counter = 0
            for info, rows in hist.groupby(['text_id', 'word_id']):
                item_id.append(item_counter)
                item_counter += 1
        eye_tracking['id'] = item_id

    # make sure words look like words in simulation (lowercased, without punctuation, etc)
    eye_tracking['word'] = [pre_process_string(word) for word in eye_tracking['word']]

    return eye_tracking

def get_first_pass_fixations(simulation_df:pd.DataFrame):

    """
    Gather first-pass fixation to determine word-measures that are based on first-pass fixations, e.g. skipping, first fixation, single fixation and gaze duration.
    First-pass fixations of a word are defined as the first fixation on that word resulted from a forward or wordskip saccade, plus all sequential refixations.
    :param simulation_output: contains the fixations of the model simulation(s)
    :return: a pandas dataframe with only the fixations belonging to first-pass
    """

    fix_counter = []
    # reset fixation counter given first text word being excluded (not in eye-tracking, if PROVO)
    for sim_id, sim_hist in simulation_df.groupby('simulation_id'):
        for text_id, text_hist in sim_hist.groupby('text_id'):
            fix_counter.extend([i for i in range(0, len(text_hist['fixation_counter'].tolist()))])
    simulation_df['fixation_counter'] = fix_counter

    first_pass_indices = []
    for sim_id, sim_hist in simulation_df.groupby('simulation_id'):
        for text_id, text_hist in sim_hist.groupby('text_id'):
            for word, fix_hist in text_hist.groupby('word_id'):
                # make sure the eyes have NOT moved to beyond the word before fixating it for the first time
                if fix_hist['fixation_counter'].tolist()[0] == 0 or all(word > previous_word for previous_word in
                                                                        text_hist['word_id'].tolist()[:fix_hist['fixation_counter'].tolist()[0]]):
                    # check each fixation on the word
                    for i, fix in fix_hist.iterrows():
                        # if fixation was not the result of a regression
                        if fix['saccade_type'] != 'regression':
                            # add to first-pass if first fixation on word
                            if fix['fixation_counter'] == fix_hist['fixation_counter'].tolist()[0]:
                                first_pass_indices.append(i)
                            # add to first-pass the following fixations which are sequential to first fixation
                            elif i == first_pass_indices[-1] + 1:
                                first_pass_indices.append(i)

    first_pass = simulation_df.filter(items=first_pass_indices, axis=0)

    return first_pass

def get_text_words(stimuli):

    words_in_text = defaultdict(list)
    for word_info, rows in stimuli.groupby(['text_id', 'word_id']):
        words_in_text[int(word_info[0])].append(rows['word'].tolist()[0])
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
                skipped_words['simulation_id'].append(sim_id)
                skipped_words['text_id'].append(text_id)
                skipped_words['word_id'].append(word_id)
                skipped_words['word'].append(word)
                if word_id not in list(text_hist['word_id'].unique()):
                    skipped_words['skip'].append(1)
                else:
                    skipped_words['skip'].append(0)

    skipped_added = pd.DataFrame.from_dict(skipped_words)

    return skipped_added

def get_single_fix_words(first_pass):

    single_fix = defaultdict(list)

    for word_info, hist in first_pass.groupby(['simulation_id', 'text_id', 'word_id']):
        single_fix['simulation_id'].append(word_info[0])
        single_fix['text_id'].append(word_info[1])
        single_fix['word_id'].append(word_info[2])
        if len(hist) == 1:
            single_fix['single_fix'].append(1)
            single_fix['single_fix_duration'].append(hist['fixation_duration'].tolist()[0])
        else:
            single_fix['single_fix'].append(0)
            single_fix['single_fix_duration'].append(None)

    single_fix = pd.DataFrame.from_dict(single_fix)

    return single_fix

def get_regressions_in(simulation_output):

    regressions = defaultdict(list)

    for word_info, hist in simulation_output.groupby(['simulation_id', 'text_id', 'word_id']):
        regressions['simulation_id'].append(word_info[0])
        regressions['text_id'].append(word_info[1])
        regressions['word_id'].append(word_info[2])
        if 'regression' in hist['saccade_type'].tolist():
            regressions['regression_in'].append(1)
        else:
            regressions['regression_in'].append(0)

    regressions_in = pd.DataFrame.from_dict(regressions)

    return regressions_in

def aggregate_fixations_per_word(simulation_output, first_pass, stimuli, measures):

    """
    Transform fixation data into word-centred data, where each row is a word in each text/trial in each simulation.
    Add columns with word-level eye movement measures
    :param simulation_output:
    :param first_pass:
    :param stimuli:
    :param measures:
    :return: dataframe with fixations aggregated per word per simulation
    """

    results_per_word = dict()

    for measure in measures:

        if measure == 'skip':
            results_per_word['skip'] = get_skipped_words(first_pass, get_text_words(stimuli))

        elif measure == 'single_fix':
            results_per_word['single_fix'] = get_single_fix_words(first_pass)[['simulation_id', 'text_id', 'word_id', 'single_fix']]

        elif measure == 'single_fix_duration':
            results_per_word['single_fix_duration'] = get_single_fix_words(first_pass)[['simulation_id', 'text_id', 'word_id', 'single_fix_duration']]

        elif measure == 'first_fix_duration':
            results_per_word['first_fix_duration'] = first_pass.loc[first_pass.groupby(
                ['simulation_id', 'text_id', 'word_id']).apply(lambda x: x.index[0]).values, :].reset_index().rename(
                columns={'fixation_duration': 'first_fix_duration'})

        elif measure == 'gaze_duration':
            results_per_word['gaze_duration'] = first_pass.groupby(['simulation_id', 'text_id', 'word_id'])[
                ['fixation_duration']].sum().reset_index().rename(columns={'fixation_duration': 'gaze_duration'})

        elif measure == 'total_reading_time':
            results_per_word['total_reading_time'] = simulation_output.groupby(['simulation_id', 'text_id', 'word_id'])[
                ['fixation_duration']].sum().reset_index().rename(columns={'fixation_duration': 'total_reading_time'})

        elif measure == 'regression_in':
            results_per_word['regression_in'] = get_regressions_in(simulation_output)

        elif measure == 'refixation':
            pass
            # refixation = first_pass.groupby(['word_id']).filter(lambda x:len(x)>1)

    # merge all
    result_columns = defaultdict(list)
    for item, hist in stimuli.groupby(['text_id', 'word_id']):
        for simulation_id in simulation_output['simulation_id'].unique():
            result_columns['simulation_id'].append(simulation_id)
            result_columns['text_id'].append(item[0])
            result_columns['word_id'].append(item[1])
            result_columns['word'].append(hist['word'].tolist()[0])
            for measure in results_per_word.keys():
                measure_value = np.nan
                item_info = f"simulation_id=={simulation_id} & text_id=={item[0]} & word_id=={item[1]}"
                cell = results_per_word[measure].query(item_info)[measure]
                if len(cell) > 0:
                    measure_value = cell.item()
                result_columns[measure].append(measure_value)

    results = pd.DataFrame(result_columns)

    return results

def get_word_factors(pm, eye_movements_df):

    if 'predictability' in pm.fixed_factors:
        pred_map = get_pred_dict(pm,None)
        pred_column = []
        for i, item in eye_movements_df.iterrows():
            pred_value = 0.0
            if str(item['text_id']) in pred_map.keys():
                if str(item['word_id']) in pred_map[str(item['text_id'])].keys():
                    word = item['word'].strip()
                    if word in pred_map[str(item['text_id'])][str(item['word_id'])]['predictions'].keys():
                        pred_value = pred_map[str(item['text_id'])][str(item['word_id'])]['predictions'][word]
            pred_column.append(pred_value)
        eye_movements_df['predictability'] = pred_column

    if 'frequency' in pm.fixed_factors:
        freq_map = get_word_freq(pm,None)
        freq_column = []
        for i, item in eye_movements_df.iterrows():
            freq_value = 0.0
            word = item['word'].strip()
            if word in freq_map.keys():
                freq_value = freq_map[word]
            freq_column.append(freq_value)
        eye_movements_df['frequency'] = freq_column

    if 'length' in pm.fixed_factors:
        length_column = []
        for i, item in eye_movements_df.iterrows():
            word = item['word'].strip()
            length_column.append(len(word))
        eye_movements_df['length'] = length_column

    return eye_movements_df

def drop_nan_values(true_values:pd.Series, simulated_values:pd.Series):

    values = defaultdict(list)
    counter = 0
    for true_value, simulated_value in zip(true_values, simulated_values):
        if not true_values.isnull().tolist()[counter] and not simulated_values.isnull().tolist()[counter]:
            values['true'].append(true_value)
            values['pred'].append(simulated_value)
        counter += 1
    return values

def word_recognition_acc_to_factor(accuracy, word_factor, recog_cycles):

    count, acc_dict = defaultdict(dict), defaultdict(dict)

    for factor_value, recog, cycle in zip(word_factor, accuracy, recog_cycles):

        if factor_value != -1:

            factor_value = str(round(factor_value))
            cycle = int(cycle)

            if factor_value in count.keys():
                count[factor_value]['recog_count'] += recog
                if cycle >= 0:
                    if 'cycle_count' in count[factor_value].keys():
                        count[factor_value]['cycle_count'] += cycle
                    else:
                        count[factor_value]['cycle_count'] = cycle
            else:
                count[factor_value] = {'recog_count': recog}
                if cycle >= 0:
                    count[factor_value]['cycle_count'] = cycle

    for factor_value in count.keys():
        acc_dict[factor_value]['mean_acc'] = count[factor_value]['recog_count'] / len(list(filter(lambda x: round(x) == int(factor_value), word_factor)))
        if 'cycle_count' in count[factor_value].keys():
            acc_dict[factor_value]['mean_cycle'] = count[factor_value]['cycle_count'] / len(list(filter(lambda x: round(x) == int(factor_value), word_factor)))

    return acc_dict

def word_recognition_acc(fixations, pm):

    acc, lengths, freqs, recog_cycles = [], [], [], []
    freq_map = get_word_freq(pm, None)

    for text_id, text_fix in fixations.groupby('text_id'):

        recognized_words = text_fix['recognized_words'].tolist()[-1].replace('[','').replace(']','').replace(',','').replace("'","").split()
        text_words = text_fix['trial_words'].tolist()[-1].replace('[','').replace(']','').replace(',','').replace("'","").split()

        assert len(recognized_words) == len(text_words)

        acc.extend([1 if recognized_word == text_word else 0 for recognized_word, text_word in zip(recognized_words, text_words)])
        recog_cycles.extend(text_fix['cycle_of_recognition'].tolist()[-1].replace('[', '').replace(']', '').replace(',','').replace("'", "").split())

        for text_word in text_words:
            lengths.append(len(text_word))
            if text_word in freq_map.keys():
                freqs.append(freq_map[text_word])
            else:
                freqs.append(-1)

    # recog acc and recog speed in relation to a word factor
    len_acc = word_recognition_acc_to_factor(acc, lengths, recog_cycles)
    freq_acc = word_recognition_acc_to_factor(acc, freqs, recog_cycles)

    return acc, len_acc, freq_acc

def fit_mixed_effects(parameters, true, predicted, output_filepath):
    
    # generalized mixed effects model doc: https://www.statsmodels.org/stable/generated/statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM.from_formula.html#statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM.from_formula
    # linear mixed effects doc: https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.from_formula.html

    predicted_dict = defaultdict(dict)
    for text, hist in predicted.groupby(['text_id','word_id']):
        predicted_dict[text[0]][text[1]] = dict()
        for measure in parameters.evaluation_measures:
            predicted_dict[text[0]][text[1]][measure] = hist[measure].tolist()[0]

    simulated_measures = defaultdict(list)
    for item, hist in true.groupby(['participant_id', 'text_id', 'word_id']):
        for measure in parameters.evaluation_measures:
            pred_value = predicted_dict[item[1]][item[2]][measure]
            simulated_measures[measure].append(pred_value)

    # linear mixed effects models to determine if the simulation data can predict the human data
    for measure in parameters.evaluation_measures:
        # # merge pred and true
        # # makes sure predicted means are repeated for each participant in human data
        # predicted_measure = []
        # print('start merging')
        # for item, hist in true.groupby(['participant_id', 'text_id', 'word_id']):
        #     pred_value = predicted.query(f"text_id=={item[1]} & word_id=={item[2]}")[measure]
        #     predicted_measure.append(pred_value.item())
        true[f'simulated_{measure}'] = simulated_measures[measure]
        # print('end merging')
        data_measure = true[['participant_id', 'id', 'text_id', 'word_id', measure, f'simulated_{measure}']]
        # drop nan values as statsmodels cannot handle nan values and throws error in case there is any
        data_measure = data_measure.dropna()
        # round decimals to whole numbers
        data_measure[measure].astype(int)
        data_measure[f'simulated_{measure}'].astype(int)
        # for now, td floats to whole number so save time, delete later
        if parameters.prediction_flag == 'cloze' and measure in ['single_fix_duration', 'skip', 'single_fix', 'first_fix_duration', 'gaze_duration ']:
            continue
        print('start stats model')
        # in case of binary dependent variable (eye-movement measure)
        if {int(value) for value in data_measure[measure].unique()} == {1, 0}:
            random = {"participant_id": "0+C(participant_id)",
                       "id": "0+C(id)"}
            stats_model = sm.BinomialBayesMixedGLM.from_formula(f"{measure} ~ simulated_{measure}",
                                                                vc_formulas=random,
                                                                data=data_measure)
            stats_result = stats_model.fit_map(method='BFGS')
            print(stats_result.summary())
            # stats_result.save(output_filepath.replace('simulation_', f'glm_{measure}_').replace('.csv', '.pkl'))
            with open(output_filepath.replace('simulation_', f'glm_{measure}_').replace('.csv', '.txt').replace('model_output','analysed'), 'w') as fh:
                fh.write(stats_result.summary().as_text())
        else:
            # stats_model = smf.mixedlm(f"{measure} ~ simulated_{measure}",data_measure,groups=data_measure['participant_id'])
            # from statsmodel doc: To include crossed random effects in a model,
            # it is necessary to treat the entire dataset as a single group.
            # The variance components arguments to the model can then be used to define models
            # with various combinations of crossed and non-crossed random effects.
            data_measure["group"] = 1
            # add independent random intercept for each vc_formula key
            random = {"participant_id": "0+C(participant_id)",
                       "id": "0+C(id)"}
            stats_model = sm.MixedLM.from_formula(f"{measure} ~ simulated_{measure}",
                                                  groups="group",
                                                  vc_formula=random,
                                                  data=data_measure)
            # stats_model = sm.MixedLM.from_formula(f"{measure} ~ simulated_{measure}",
            #                                       groups='participant_id',
            #                                       re_formula="1",
            #                                       vc_formula={"text_id":"0+C(text_id)",
            #                                                   "word_id":"0+C(word_id)"},
            #                                       data=data_measure)
            stats_result = stats_model.fit(method=["lbfgs"])
            print(stats_result.summary())
           #  stats_result.save(output_filepath.replace('simulation_', f'lm_{measure}_').replace('.csv','.pkl'))
            with open(output_filepath.replace('simulation_', f'lm_{measure}_').replace('.csv', '.txt').replace('model_output','analysed'), 'w') as fh:
                fh.write(stats_result.summary().as_text())
        print('end stats model')

def scale_human_durations(data_log, parameters_list):

    all_measures = [measure for parameters in parameters_list for measure in parameters.evaluation_measures]
    for data_name, data in data_log.items():
        if 'eye_tracking' in data_name.lower():
            for measure in all_measures:
                if measure in ['total_reading_time', 'first_fixation_duration', 'gaze_duration']:
                    # check how many ob1 reading cycles the duration corresponds and multiply the number of cycles with 25
                    data[measure] = data[measure].apply(lambda x: int(x / 25) * 25 if not np.isnan(x) else x)
    return data_log

def merge_human_and_simulation_data(data_log, parameters_list):

    all_data = []
    for data_name, data in data_log.items():
        if data_name == parameters_list[0].eye_tracking_filepath + '_mean':
            if 'provo' in data_name.lower():
                data["predictor"] = ['PROVO' for i in range(len(data))]
                data['id'] = [i for i in range(len(data))]
                all_data.append(data)
        elif 'language_model' in data_name.lower() and '_mean' in data_name.lower():
            data["predictor"] = ['language_model' for i in range(len(data))]
            data['id'] = [i for i in range(len(data))]
            all_data.append(data)
        elif 'cloze' in data_name.lower() and '_mean' in data_name.lower():
            data['predictor'] = ['cloze' for i in range(len(data))]
            data['id'] = [i for i in range(len(data))]
            all_data.append(data)
    data = pd.concat(all_data, axis=0).reset_index()

    return data

def plot_fixed_factor_vs_eye_movement(data, fixed_factors, measures, results_filepath):

    results_dir = os.path.dirname(results_filepath).replace('model_output', 'analysed')
    for fixed_factor in fixed_factors:
        for measure in measures:
            plot = sb.relplot(data=data, x=measure, y=fixed_factor, hue='predictor', kind='line')
            filepath = f"{results_dir}/plot_{fixed_factor}_{measure}.png"
            plot.figure.savefig(filepath)

def process_eye_tracking_data(parameters, data_log, simulation_output):

    processed_eye_tracking_path = parameters.eye_tracking_filepath.replace('-Eyetracking_Data',
                                                                           '_eye_tracking_last_sim').replace('raw', 'processed')
    processed_mean_eye_tracking_path = parameters.eye_tracking_filepath.replace('-Eyetracking_Data',
                                                                                '_eye_tracking_last_sim_mean').replace('raw', 'processed')
    if processed_mean_eye_tracking_path in data_log.keys():
        # true_eye_movements = data_log[parameters.eye_tracking_filepath]
        mean_true_eye_movements = data_log[processed_mean_eye_tracking_path]
    else:
        # read in eye_tracking data for comparing simulation with observed measures
        # encoding = chardet.detect(open(parameters.eye_tracking_filepath, "rb").read())['encoding']
        eye_tracking = pd.read_csv(parameters.eye_tracking_filepath, encoding="ISO-8859-1")
        # pre-process eye-tracking data to the format needed
        true_eye_movements = pre_process_eye_tracking(eye_tracking, parameters.eye_tracking_filepath, parameters.stim)
        # only including texts that were included in simulations
        text_ids = [int(text_id) for text_id in simulation_output['text_id'].unique().tolist()]
        true_eye_movements = true_eye_movements[true_eye_movements['text_id'].isin(text_ids)]
        # save out pre-processed data
        true_eye_movements.to_csv(processed_eye_tracking_path, sep='\t')
        data_log[processed_eye_tracking_path] = true_eye_movements
        # get word level measures from eye_tracking, averaged over participants
        mean_true_eye_movements = true_eye_movements.groupby(['text_id', 'word_id', 'word']) \
            [parameters.evaluation_measures].mean().reset_index()
        # round values
        mean_true_eye_movements[parameters.evaluation_measures] = mean_true_eye_movements[
            parameters.evaluation_measures].round(3)
        # save out averaged data
        mean_true_eye_movements.to_csv(processed_mean_eye_tracking_path, sep='\t', index_label='id')
        data_log[processed_mean_eye_tracking_path] = mean_true_eye_movements

    return mean_true_eye_movements, data_log

def process_simulation_data(parameters, data_log, simulation_output, output_filepath, mean_true_eye_movements):

    analysed_simulation_output_path = output_filepath.replace('model_output', 'analysed').replace('simulation_',
                                                                                                  f'simulation_eye_movements_')
    analysed_mean_simulation_output_path = output_filepath.replace('model_output', 'analysed').replace('simulation_',
                                                                                                       f'simulation_eye_movements_mean_')
    stimuli = mean_true_eye_movements[['text_id', 'word_id', 'word']]
    # first aggregate fixations per word, keeping simulation level
    predicted_eye_movements = aggregate_fixations_per_word(simulation_output,
                                                           get_first_pass_fixations(simulation_output),
                                                           stimuli,
                                                           parameters.evaluation_measures)

    # ----------- get word factors (e.g. frequency, length, predictability) -----------
    # predicted_eye_movements = get_word_factors(parameters, predicted_eye_movements, parameters.fixed_factors)
    # save it
    dir = os.path.dirname(analysed_simulation_output_path)
    if not os.path.exists(dir): os.makedirs(dir)
    predicted_eye_movements.to_csv(analysed_simulation_output_path, sep='\t', index_label='id')
    data_log[analysed_simulation_output_path] = predicted_eye_movements
    # then average over simulations to get the means
    # variables = parameters.evaluation_measures.extend(parameters.fixed_factors)
    variables = parameters.evaluation_measures
    mean_predicted_eye_movements = predicted_eye_movements.groupby(['text_id', 'word_id', 'word']) \
        [variables].mean().reset_index()
    # add fixed factors
    mean_predicted_eye_movements = get_word_factors(parameters, mean_predicted_eye_movements)
    # round values
    mean_predicted_eye_movements[variables] = mean_predicted_eye_movements[variables].round(3)
    # save it
    mean_predicted_eye_movements.to_csv(analysed_mean_simulation_output_path, sep='\t', index_label='id')
    data_log[analysed_mean_simulation_output_path] = mean_predicted_eye_movements

    return mean_predicted_eye_movements, predicted_eye_movements, data_log

def compute_root_mean_squared_error(true_values:list, simulated_values:list):

    # root mean squared error measures the average difference between values predicted by the model
    # and the eye-tracking values. It provides an estimate of how well the model was able to predict the
    # eye-tracking value.

    diff = np.subtract(simulated_values, true_values)

    # if normalize:
    # # normalize difference: subtract the min difference from the difference and divide it by the difference between max and min differences
    #     diff = np.divide((np.subtract(diff, min(diff))), np.subtract(max(diff), min(diff)))

    # # another way: first normalize values with min-max scaler, then compute difference
    # norm_sim_values = np.divide((np.subtract(simulated_values, min(true_values))), np.subtract(max(true_values), min(true_values)))
    # norm_true_values = np.divide((np.subtract(true_values, min(true_values))), np.subtract(max(true_values), min(true_values)))
    # norm_diff = np.subtract(norm_sim_values, norm_true_values)

    # or standardize values: subtract value by the mean and divide by standard deviation,
    # but Gaussian distribution is assumed
    norm_sim_values = np.divide(np.subtract(simulated_values, np.mean(true_values)), np.std(true_values))
    norm_true_values = np.divide(np.subtract(true_values, np.mean(true_values)), np.std(true_values))
    norm_diff = np.subtract(norm_sim_values, norm_true_values)
    # # standardize using median
    # norm_sim_values = np.divide(np.subtract(simulated_values, np.median(true_values)), np.median(np.absolute(true_values - np.median(true_values))))
    # norm_true_values = np.divide(np.subtract(true_values, np.median(true_values)), np.median(np.absolute(true_values - np.median(true_values))))
    # diff = np.subtract(norm_sim_values, norm_true_values)

    return math.sqrt(np.square(diff).mean()), math.sqrt(np.square(norm_diff).mean()), norm_sim_values, norm_true_values,

def compute_error(measures, true, pred):

    mean2errors = defaultdict(list)
    for measure in measures:
        # excluding words with nan value, e.g. skipped words of prob 1. Should words that are always skipped be included in the equation?
        values = drop_nan_values(true[measure], pred[measure])
        assert len(values['pred']) > 0, print(measure, values['pred'])
        assert len(values['pred']) == len(values['true']), print(measure, len(values['pred']), len(values['true']))
        mean2error, norm_mean2error, norm_sim, norm_true = compute_root_mean_squared_error(values['true'], values['pred'])
        mean2errors['eye_tracking_measure'].append(measure)
        mean2errors['true_mean'].append(np.round(np.nanmean(true[measure]), 3))
        # mean2errors['norm_true_mean'].append(np.round(np.mean(norm_true), 3))
        mean2errors['min_true_mean'].append(np.round(np.nanmin(true[measure]),3))
        mean2errors['max_true_mean'].append(np.round(np.nanmax(true[measure]),3))
        mean2errors['min_simulated_mean'].append(np.round(np.nanmin(pred[measure]),3))
        mean2errors['max_simulated_mean'].append(np.round(np.nanmax(pred[measure]),3))
        mean2errors['norm_simulated_mean'].append(np.round(np.mean(norm_sim),3))
        mean2errors['simulated_mean'].append(np.round(np.nanmean(pred[measure]), 3))
        mean2errors['raw_difference'].append(np.subtract((np.round(np.nanmean(pred[measure]), 3)),(np.round(np.nanmean(true[measure]), 3))))
        mean2errors['mean_squared_error'].append(round(mean2error, 3))
        mean2errors['norm_mean_squared_error'].append(round(norm_mean2error, 3))

    average_norm_error = np.mean(mean2errors['norm_mean_squared_error'])
    mean2errors['eye_tracking_measure'].append("MEAN")
    mean2errors['true_mean'].append(None)
    # mean2errors['norm_true_mean'].append(None)
    mean2errors['min_true_mean'].append(None)
    mean2errors['max_true_mean'].append(None)
    mean2errors['min_simulated_mean'].append(None)
    mean2errors['max_simulated_mean'].append(None)
    mean2errors['norm_simulated_mean'].append(None)
    mean2errors['simulated_mean'].append(None)
    mean2errors['raw_difference'].append(None)
    mean2errors['mean_squared_error'].append(None)
    mean2errors['norm_mean_squared_error'].append(average_norm_error.round(3))

    mean2error_df = pd.DataFrame(mean2errors)

    return mean2error_df

def create_new_directory(filepath, new_folder):

    dir = f"{os.path.dirname(filepath)}/{new_folder}"
    filename = os.path.basename(filepath)
    if not os.path.isdir(dir): os.makedirs(dir)
    filepath = f"{dir}/{filename}"

    return filepath

def compute_all_error(parameters, output_filepath, mean_true_eye_movements, mean_predicted_eye_movements, predicted_eye_movements, data_log, verbose):

    mean2error_df = compute_error(parameters.evaluation_measures,
                                  mean_true_eye_movements,
                                  mean_predicted_eye_movements)
    filepath = output_filepath.replace('model_output', 'analysed').replace('simulation_', f'RM2E_mean_eye_movements_')
    filepath = create_new_directory(filepath, 'RM2E')
    mean2error_df.to_csv(filepath, sep='\t', index=False)
    data_log[filepath] = mean2error_df
    if verbose:
        print(mean2error_df.head(len(parameters.evaluation_measures) + 1))

    # ----------- mean square error per simulation -----------
    error_dfs = []
    for sim, sim_info in predicted_eye_movements.groupby('simulation_id'):
        df = compute_error(parameters.evaluation_measures,
                           mean_true_eye_movements,
                           sim_info)
        df['simulation_id'] = [sim for i in df.eye_tracking_measure.tolist()]
        error_dfs.append(df)
    error_df = pd.concat(error_dfs)
    error_df = error_df.pivot_table('norm_mean_squared_error', ['simulation_id'], 'eye_tracking_measure')
    #cerror_df = error_df.pivot_table('mean_squared_error', ['simulation_id'], 'eye_tracking_measure')
    error_df = error_df.sort_values('simulation_id')
    filepath = output_filepath.replace('model_output', 'analysed').replace('simulation_', f'RM2E_eye_movements_')
    filepath = create_new_directory(filepath, 'RM2E')
    error_df.to_csv(filepath, sep='\t')
    data_log[filepath] = error_df

    return data_log

def compute_word_recog_acc(simulation_output_all, simulation_output, parameters, output_filepath, verbose):

    all_acc, sim_ids_all = [], []
    len_accs, freq_accs = dict(), dict()
    sim_count_len = dict()
    sim_count_freq = dict()

    for sim_id, fixations in simulation_output_all.groupby('simulation_id'):
        acc, len_acc, freq_acc = word_recognition_acc(fixations, parameters)
        all_acc.append(round(sum(acc) / len(acc), 3))
        sim_ids_all.append(sim_id)

        for length in len_acc.keys():
            if length in len_accs.keys():
                len_accs[length]['mean_acc'] += len_acc[length]['mean_acc']
                if 'mean_cycle' in len_acc[length].keys():
                    if 'mean_cycle' in len_accs[length].keys():
                        len_accs[length]['mean_cycle'] += len_acc[length]['mean_cycle']
                        sim_count_len[length] += 1
                    else:
                        len_accs[length]['mean_cycle'] = len_acc[length]['mean_cycle']
                        sim_count_len[length] = 1
            else:
                len_accs[length] = {'mean_acc': len_acc[length]['mean_acc']}
                if 'mean_cycle' in len_acc[length].keys():
                    len_accs[length]['mean_cycle'] = len_acc[length]['mean_cycle']
                    sim_count_len[length] = 1

        for freq in freq_acc.keys():
            if freq in freq_accs.keys():
                freq_accs[freq]['mean_acc'] += freq_acc[freq]['mean_acc']
                if 'mean_cycle' in freq_acc[freq].keys():
                    if 'mean_cycle' in freq_accs[freq].keys():
                        freq_accs[freq]['mean_cycle'] += freq_acc[freq]['mean_cycle']
                        sim_count_freq[freq] += 1
                    else:
                        freq_accs[freq]['mean_cycle'] = freq_acc[freq]['mean_cycle']
                        sim_count_freq[freq] = 1
            else:
                freq_accs[freq] = {'mean_acc': freq_acc[freq]['mean_acc']}
                if 'mean_cycle' in freq_acc[freq].keys():
                    freq_accs[freq]['mean_cycle'] = freq_acc[freq]['mean_cycle']
                    sim_count_freq[freq] = 1

    for length in len_accs.keys():
        len_accs[length]['mean_acc'] = round(len_accs[length]['mean_acc'] / len(sim_ids_all), 3)
        if 'mean_cycle' in len_accs[length].keys():
            len_accs[length]['mean_cycle'] = round(len_accs[length]['mean_cycle'] / sim_count_len[length], 3)
    for freq in freq_accs.keys():
        freq_accs[freq]['mean_acc'] = round(freq_accs[freq]['mean_acc'] / len(sim_ids_all), 3)
        if 'mean_cycle' in freq_accs[freq].keys():
            freq_accs[freq]['mean_cycle'] = round(freq_accs[freq]['mean_cycle'] / sim_count_freq[freq], 3)

    all_acc.append(round(sum(all_acc) / len(all_acc), 3))
    sim_ids_all.append('MEAN')
    recog_acc_df = pd.DataFrame({'simulation_id': sim_ids_all, 'word_recognition_accuracy': all_acc})
    filepath = output_filepath.replace('model_output', 'analysed').replace('simulation_', f'word_recognition_acc_')
    filepath = create_new_directory(filepath, 'word_rcg_acc')
    recog_acc_df.to_csv(filepath, sep='\t', index=False)

    recog_acc_df_len = pd.DataFrame({'length': [int(length) for length in len_accs.keys()],
                                     'word_recognition_accuracy': [item['mean_acc'] for length, item in
                                                                   len_accs.items()],
                                     'word_recognition_cycle': [
                                         item['mean_cycle'] if 'mean_cycle' in item.keys() else None for length, item in
                                         len_accs.items()]})
    filepath = output_filepath.replace('model_output', 'analysed').replace('simulation_',
                                                                           f'word_recognition_acc_length_')
    filepath = create_new_directory(filepath, 'word_rcg_acc')
    recog_acc_df_len = recog_acc_df_len.sort_values('length')
    recog_acc_df_len.to_csv(filepath, sep='\t', index=False)

    recog_acc_df_freq = pd.DataFrame({'freq': [int(freq) for freq in freq_accs.keys()],
                                      'word_recognition_accuracy': [item['mean_acc'] for freq, item in
                                                                    freq_accs.items()],
                                      'word_recognition_cycle': [
                                          item['mean_cycle'] if 'mean_cycle' in item.keys() else None for freq, item in
                                          freq_accs.items()]})
    filepath = output_filepath.replace('model_output', 'analysed').replace('simulation_', f'word_recognition_acc_freq_')
    filepath = create_new_directory(filepath, 'word_rcg_acc')
    recog_acc_df_freq = recog_acc_df_freq.sort_values('freq')
    recog_acc_df_freq.to_csv(filepath, sep='\t', index=False)

    if verbose: print(recog_acc_df.head(len(simulation_output['simulation_id'].unique()) + 1))


def plot_word_measures(data, measures, results_filepath):

    results_dir = os.path.dirname(results_filepath).replace('model_output', 'analysed')
    for measure in measures:
        # plot item level mean
        data['id'] = data['id'].apply(lambda x: str(x))
        plot = sb.relplot(data=data, x='id', y=measure, hue='predictor', kind='scatter')
        filepath = f"{results_dir}/plot_item_{measure}.png"
        plot.figure.savefig(filepath)
        # plot experiment level mean
        # data_mean = data[['predictor',measure]].groupby('predictor').mean().reset_index()
        plot = sb.barplot(data=data, x="predictor", y=measure)
        filepath = f"{results_dir}/plot_{measure}.png"
        plot.figure.savefig(filepath)

def plot_error(error_values, conditions, measure, filepath):

    plt.figure()
    plot = sb.barplot(y=error_values, x=conditions)
    plot.set(ylabel=measure)
    dir = os.path.dirname(filepath)
    filepath = f"{dir}/plot_{measure}_error.png"
    plot.figure.savefig(filepath)
    plt.close()

def plot_raw_measures(raw_values, conditions, measure, filepath):

    plt.figure()
    plot = sb.barplot(x=conditions, y=raw_values, errorbar="sd",
                     errwidth=0.5) # TODO upate version seaborn to use err_kws argument
    # err_kws={"alpha": 0.2, "linewidth": 0.5}
    plot.set(ylabel=measure)
    plot.bar_label(plot.containers[0])
    filepath = filepath.replace('.csv', f'_plot_{measure}_raw.png')
    plot.figure.savefig(filepath)
    plt.close()

def test_difference(sim_values1, sim_values2, filepath):

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
    # "The T-statistic measures whether the average score differs significantly across samples.
    # It is calculated as np.mean(a - b)/se, where se is the standard error.
    # Therefore, the T-statistic will be positive when the sample mean of a - b is greater than zero
    # and negative when the sample mean of a - b is less than zero"

    assert len(sim_values1) == len(sim_values2)

    # # test if distribution is normal
    # if len(sim_values1) > 3:
    #     print(stats.shapiro(sim_values1))
    # if len(sim_values2) > 3:
    #     print(stats.shapiro(sim_values2))
    # print(measure)
    # print(sim_values1, sim_values2)
    # # test if difference in mean2error is significant between conditions
    # test = stats.ttest_rel(sim_values2, sim_values2)
    # t_test = {'test': ['t-statistics', 'p-value', 'degrees of freedom', 'confidence interval'],
    #          'result': [test.statistic, test.pvalue, test.df, test.confidence_interval()]}
    # print(test)

    # AL: since no normal distribution, try Wilcoxon T-test
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
    # It is a non-parametric version of the paired T-test
    test = stats.wilcoxon(sim_values1, sim_values2)
    t_test = {'test': ['t-statistics', 'p-value'],
              'result': [test.statistic, test.pvalue]}
    # Wilcoxon statistic: the sum of the ranks of the differences above zero
    df = pd.DataFrame.from_dict(t_test)
    print(df)
    print(filepath)
    df.to_csv(filepath, sep='\t', index=False)

def compare_conditions(measures, data_log):

    conditions = ['baseline', 'cloze', 'gpt2', 'llama']

    for measure in measures:
        lists_to_compare = dict()
        for condition in conditions:
            for data_name, data in data_log.items():
                if condition in data_name and 'RM2E' in data_name and 'mean' not in data_name:
                    results_dir = os.path.dirname(data_name).replace('model_output', 'analysed')
                    lists_to_compare[condition] = data[measure].tolist()

        for combi in itertools.combinations(lists_to_compare.keys(), 2):
            # filepath = parameters_list[0].results_filepath.replace('cloze', '').replace('gpt2', '').replace('llama', '')
            filepath = f'{results_dir}/t-test_{measure}_{combi}.csv'
            test_difference(lists_to_compare[combi[0]], lists_to_compare[combi[1]], filepath)

    # add average over measures
    lists_to_compare = dict()
    for condition in conditions:
        for data_name, data in data_log.items():
            if condition in data_name and 'RM2E' in data_name and 'mean' not in data_name:
                errors_per_measure, mean_rmse = [], []
                for measure in measures:
                    errors_per_measure.append(data[measure].tolist())
                for simulation_id in range(len(errors_per_measure[0])):
                    mean_rmse.append(np.mean([measure_errors[simulation_id] for measure_errors in errors_per_measure]))
                lists_to_compare[condition] = mean_rmse
    for combi in itertools.combinations(lists_to_compare.keys(), 2):
        print(combi)
        # filepath = parameters_list[0].results_filepath.replace('cloze', '').replace('gpt2', '').replace('llama', '')
        filepath = f'{results_dir}/t-test_mean_{combi}.csv'
        test_difference(lists_to_compare[combi[0]], lists_to_compare[combi[1]], filepath)

def plot_RMSE(eye_measures, data_log, conditions, weights):

    rmse_per_condition = defaultdict()

    for predictor in conditions:
        rmse_per_measure = defaultdict()
        for measure in eye_measures:
            rmse_per_weight = dict()
            for weight in weights:
                for data_name, data in data_log.items():
                    if predictor in data_name and weight in data_name and 'RM2E' in data_name and 'mean' not in data_name:
                    # if predictor in data_name and 'RM2E' in data_name:
                        results_dir = os.path.dirname(data_name).replace('model_output', 'analysed').replace('RM2E', 'plots')
                        # rmse = data.loc[data['eye_tracking_measure'] == measure]['norm_mean_squared_error'].tolist()[0]
                        rmse = data[measure].tolist()
                        # rmse_per_weight[weight] = float(rmse)
                        if predictor == 'baseline':
                            rmse_per_weight['baseline'] = rmse
                        else:
                            rmse_per_weight[weight] = rmse
            rmse_per_measure[measure] = rmse_per_weight
        rmse_per_condition[predictor] = rmse_per_measure
        # add mean over eye movement measures
        rmse_per_weight = dict()
        if predictor == 'baseline':
            rmse_all_measures, mean_rmse = [], []
            for measure in eye_measures:
                rmse_all_measures.append(rmse_per_condition[predictor][measure]['baseline'])
            for simulation_id in range(len(rmse_all_measures[0])):
                mean_rmse.append(np.mean([measure_errors[simulation_id] for measure_errors in rmse_all_measures]))
            rmse_per_weight['baseline'] = mean_rmse
            rmse_per_condition[predictor]['mean'] = rmse_per_weight
        else:
            for weight in weights:
                rmse_all_measures, mean_rmse = [], []
                for measure in eye_measures:
                    rmse_all_measures.append(rmse_per_condition[predictor][measure][weight])
                for simulation_id in range(len(rmse_all_measures[0])):
                    mean_rmse.append(np.mean([measure_errors[simulation_id] for measure_errors in rmse_all_measures]))
                rmse_per_weight[weight] = mean_rmse
            rmse_per_condition[predictor]['mean'] = rmse_per_weight

    predictors, rmse_scores, weights, measures = [], [], [], []
    weight_mapping = {'0.05': 'low',
                      '0.1': 'medium',
                      '0.2': 'high',
                      'baseline': 'baseline'}
    for condition, measure_dict in rmse_per_condition.items():
        for measure, weight_dict in measure_dict.items():
            for weight, rmse_per_simulation in weight_dict.items():
                for rmse in rmse_per_simulation:
                    weights.append(weight_mapping[weight])
                    predictors.append(condition)
                    rmse_scores.append(rmse)
                    measures.append(measure.replace('_', ' '))
    df = pd.DataFrame({'condition': predictors, 'normalized error': rmse_scores, 'measure': measures, 'weights': weights})

    for measure in set(measures):
        df_measure = df.loc[df['measure'] == measure]
        # plt.rcParams.update({'font.size': 18})
        plt.figure(figsize=(13, 6))
        plot = sb.violinplot(data=df_measure, x='weights', y='normalized error', hue='condition', order=['baseline','low','medium','high'])
        sb.move_legend(plot, "upper left", bbox_to_anchor=(1, 1))
        # remove xlabel to avoid crowdness in x axis
        plot.set(xlabel=None)
        # plot.set(title=f'Error between {measure} in each model simulation and in eye-tracking data')
        if not os.path.isdir(results_dir): os.makedirs(results_dir)
        plot.figure.savefig(f'{results_dir}/plot_RMSE_{measure}.png')
        plt.close()

    predictors, rmse_scores, measures = [], [], []
    measure_name = {'skip': 'SK',
                    'first_fix_duration': 'FFD',
                    'gaze_duration': 'GD',
                    'total_reading_time': 'TRT',
                    'regression_in': 'RG'}
    for condition, measure_dict in rmse_per_condition.items():
        for measure, weight_dict in measure_dict.items():
            if measure in measure_name.keys():
                mean_rmse = []
                if weight_dict.keys():
                    # mean_rmse = sum(weight_dict.values())/len(weight_dict.keys())
                    # rmse_scores.append(mean_rmse)
                    # predictors.append(condition)
                    # measures.append(measure.replace('_', ' '))
                    for sim_id in range(len(list(weight_dict.values())[0])):
                        score = []
                        for weight in weight_dict.keys():
                            score.append(weight_dict[weight][sim_id])
                        mean_rmse.append(sum(score)/len(score))
                for rmse in mean_rmse:
                    rmse_scores.append(rmse)
                    predictors.append(condition)
                    measures.append(measure_name[measure])
    df = pd.DataFrame({'condition': predictors, 'normalized error': rmse_scores, 'measure': measures})
    plt.figure(figsize=(9, 5))
    plot = pt.RainCloud(x='measure', y='normalized error', hue='condition', data=df, orient='h', alpha =.65)
    plot.set_ylabel(None)
    plot.set_xlabel('RMSE', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend('', frameon=False)
    if not os.path.isdir(results_dir): os.makedirs(results_dir)
    plot.figure.savefig(f'{results_dir}/plot_RMSE.png', format='png', bbox_inches='tight', dpi=300)
    plt.close()

    # rmse_df = defaultdict(list)
    # for condition, measure_dict in rmse_per_condition.items():
    #     measure_rmse = dict()
    #     for measure, weight_dict in measure_dict.items():
    #         mean_rmse = []
    #         if weight_dict.keys():
    #             for sim_id in range(len(list(weight_dict.values())[0])):
    #                 score = []
    #                 for weight in weight_dict.keys():
    #                     score.append(weight_dict[weight][sim_id])
    #                 mean_rmse.append(sum(score)/len(score))
    #         measure_rmse[measure] = mean_rmse
    #     for sim_id in range(len(list(weight_dict.values())[0])):
    #         sim_rmse = [rmse_list[sim_id] for rmse_list in measure_rmse.values()]
    #         rmse_df[condition].append(sum(sim_rmse)/len(sim_rmse))
    # df = pd.DataFrame(rmse_df)
    # plt.figure(figsize=(7.5, 5))
    # plot = pt.RainCloud(data=df, bw=0.05, cut=0, orient='h')
    # plot.set_xticks(np.arange(2.2,2.7,step=.1))
    # plot.set_xlabel('RMSE', fontsize=12)
    # plt.tick_params(labelsize=12)
    # plot.figure.savefig(f'{results_dir}/plot_RMSE_mean_over_measures.png', format='png', dpi=300)
    # plt.close()

def evaluate_output (parameters_list: list, verbose=True):

    # register which human and simulated data have been analysed
    data_log = dict()

    print(f'\nEvaluating outputs...')

    # makes sure more than one experiment can be evaluated at once
    for parameters in parameters_list:

        if parameters.task_to_run == 'continuous_reading':

            output_filepath = parameters.results_filepath
            simulation_output_all = pd.read_csv(output_filepath, sep='\t')
            simulation_output_all = simulation_output_all.rename(columns={'foveal_word_index': 'word_id', 'foveal_word': 'word'})

            if verbose:
                print(f'Evaluating output in {output_filepath}')

            if 'provo' in parameters.eye_tracking_filepath.lower():
                # exclude first word of every passage (not in eye tracking -PROVO- data either)
                simulation_output = simulation_output_all[simulation_output_all['word_id'] != 0]
                # remove outliers, according to PROVO eye-tracking corpus: fixations > 80ms and < 800ms
                simulation_output = simulation_output[(simulation_output['fixation_duration'] > 80) & (simulation_output['fixation_duration'] < 800)]

            # ----------- Get word-level eye-movement measures in human data -----------
            mean_true_eye_movements, data_log = process_eye_tracking_data(parameters, data_log, simulation_output)

            # ----------- Get word-level eye-movement measures in simulation data -----------
            mean_predicted_eye_movements, predicted_eye_movements, data_log = process_simulation_data(parameters,
                                                                                                      data_log,
                                                                                                      simulation_output,
                                                                                                      output_filepath,
                                                                                                      mean_true_eye_movements)

            # ----------- Mean square error between each measure in simulation and eye-tracking -----------
            data_log = compute_all_error(parameters, output_filepath, mean_true_eye_movements,
                                        mean_predicted_eye_movements, predicted_eye_movements,
                                        data_log, verbose)

            # ----------- Word recognition accuracy -----------
            compute_word_recog_acc(simulation_output_all, simulation_output, parameters, output_filepath, verbose)

    # paths = ["../data/analysed/_2023_12_05_09-57-49/RM2E/RM2E_eye_movements_Provo_Corpus_continuous_reading_baseline_0.05.csv",
    #          "../data/analysed/_2023_12_09_13-15-47/RM2E/RM2E_eye_movements_Provo_Corpus_continuous_reading_cloze_0.2.csv",
    #          "../data/analysed/_2023_12_09_13-15-47/RM2E/RM2E_eye_movements_Provo_Corpus_continuous_reading_gpt2_0.2.csv",
    #          "../data/analysed/_2023_12_09_13-15-47/RM2E/RM2E_eye_movements_Provo_Corpus_continuous_reading_llama_0.2.csv",
    #          "../data/analysed/_2023_12_05_09-57-49/RM2E/RM2E_eye_movements_Provo_Corpus_continuous_reading_cloze_0.05.csv",
    #          "../data/analysed/_2023_12_05_09-57-49/RM2E/RM2E_eye_movements_Provo_Corpus_continuous_reading_gpt2_0.05.csv",
    #          "../data/analysed/_2023_12_05_09-57-49/RM2E/RM2E_eye_movements_Provo_Corpus_continuous_reading_llama_0.05.csv",
    #          "../data/analysed/_2023_12_07_22-32-42/RM2E/RM2E_eye_movements_Provo_Corpus_continuous_reading_cloze_0.1.csv",
    #          "../data/analysed/_2023_12_07_22-32-42/RM2E/RM2E_eye_movements_Provo_Corpus_continuous_reading_gpt2_0.1.csv",
    #          "../data/analysed/_2023_12_07_22-32-42/RM2E/RM2E_eye_movements_Provo_Corpus_continuous_reading_llama_0.1.csv"]
    #
    #
    # for path in paths:
    #     df = pd.read_csv(path, sep='\t')
    #     data_log[path] = df

    # compare_conditions(['skip', 'first_fix_duration', "gaze_duration", "total_reading_time", "regression_in"], data_log)
    # plot eye movement measures from human data vs. simulations in all pred conditions
    plot_RMSE(['skip', 'first_fix_duration', "gaze_duration", "total_reading_time", "regression_in"], data_log, ['baseline', 'gpt2', 'cloze', 'llama'], ['0.05', '0.1', '0.2'])







