import pandas as pd
import json
import seaborn as sb
from scipy.stats import binned_statistic
import numpy as np
import matplotlib.pyplot as plt
import os
import spacy
import math
from collections import defaultdict
import ptitprince as pt

def get_word_factors(word_factors, eye_movements_df, frequency_filepath=None, predictability_filepath=None):

    pred_column, freq_column, length_column, pos_tag_column, word_type_col = [], [], [], [], []

    pred_map = dict()
    if 'predictability' in word_factors:
        if predictability_filepath:
            pred_filepath = ''
            for filepath in predictability_filepath:
                for predictor in ['cloze', 'gpt2', 'llama']:
                    if predictor in filepath:
                        pred_filepath = filepath
            if pred_filepath:
                with open(pred_filepath, 'rb') as infile:
                    pred_map = json.load(infile)

    freq_map = dict()
    if 'frequency' in word_factors:
        if frequency_filepath:
            with open(frequency_filepath, 'rb') as infile:
                freq_map = json.load(infile)

    for i, item in eye_movements_df.iterrows():
        word = item['word'].strip()

        if 'frequency' in word_factors and freq_map:
            freq_value = 0.0
            if word in freq_map.keys():
                freq_value = freq_map[word]
            freq_column.append(freq_value)

        if 'predictability' in word_factors and pred_map:
            pred_value = 0.0
            if str(item['text_id']) in pred_map.keys():
                if str(item['word_id']) in pred_map[str(item['text_id'])].keys():
                    word = item['word'].strip()
                    if word in pred_map[str(item['text_id'])][str(item['word_id'])]['predictions'].keys():
                        pred_value = pred_map[str(item['text_id'])][str(item['word_id'])]['predictions'][word]
            pred_column.append(pred_value)

        if 'length' in word_factors:
            length_column.append(len(word))

        if 'pos_tag' in word_factors or 'pos_cat' in word_factors:
            doc = nlp(word)
            pos_tag = doc[0].pos_
            pos_tag_column.append(pos_tag)
            if 'pos_cat' in word_factors:
                if pos_tag in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']:
                    word_type_col.append('content')
                elif pos_tag in ['AUX', 'ADP', 'CONJ', 'PART', 'DET', 'PRON', 'SCONJ', 'CCONJ']:
                    word_type_col.append('function')
                else:
                    word_type_col.append('other') # 'NUM', 'INTJ', 'X'

    if pred_column:
        eye_movements_df['predictability'] = pred_column
    if freq_column:
        eye_movements_df['frequency'] = freq_column
    if length_column:
        eye_movements_df['length'] = length_column
    if pos_tag_column:
        eye_movements_df['pos_tag'] = pos_tag_column
        eye_movements_df['pos_cat'] = word_type_col

    return eye_movements_df

def drop_nan_values(list1, list2):

    list1_updated, list2_updated = [], []
    list1_nan = np.isnan(list1).tolist()
    list2_nan = np.isnan(list2).tolist()

    assert len(list1) == len(list2), print(len(list1), len(list2))

    for i in range(len(list1)):
        if not list1_nan[i] and not list2_nan[i]:
            list1_updated.append(list1[i])
            list2_updated.append(list2[i])

    assert len(list1_updated) == len(list2_updated), print(len(list1_updated), len(list2_updated))

    return list1_updated, list2_updated

def plot_sim_results(data_log, measures, word_variables):

    # word variable in x-axis and eye movement measure in y-axis
    for measure in measures:

        for word_variable in word_variables:

            x, y, data_type, predictors = [], [], [], []

            for predictor in ['cloze', 'gpt2', 'llama']:

                model_values, human_values, word_values = [], [], []

                for data_name, data in data_log.items():

                    if f'simulation_eye_movements_mean_Provo_Corpus_continuous_reading_{predictor}' in data_name:
                        results_dir = os.path.dirname(data_name).replace('model_output', 'analysed')
                        word_values = data[word_variable].tolist()
                        model_values = data[measure].tolist()
                    elif 'Provo_Corpus_eye_tracking_last_sim_mean.csv' in data_name or 'Provo_Corpus_eye_tracking_mean.csv' in data_name:
                        human_values = data[measure].tolist()

                if len(model_values) > 0 and len(human_values) > 0 and len(word_values) > 0:
                    for eye_movement in [model_values, human_values]:
                        x_part, y_part = [], []
                        if word_variable in ['predictability', 'frequency', 'length', 'word_id']:
                            # drop nan values
                            x_part, y_part = drop_nan_values(np.array(word_values), np.array(eye_movement))
                            y_means, edges, _ = binned_statistic(x_part, y_part, statistic='mean', bins=50)
                            x_part = edges[:-1]
                            y_part = y_means
                        if word_variable in ['pos_cat']:
                            x_part = word_values
                            y_part = eye_movement
                        x.extend(x_part)
                        y.extend(y_part)
                        if eye_movement == model_values:
                            type = 'OB1-reader'
                        else:
                            type = 'Provo'
                        data_type.extend([type for bin in x_part])
                        predictors.extend([predictor for bin in x_part])

            if len(data_type) > 0:
                measure_name = measure.replace('_', ' ')
                # seaborn lmplot function requires dataframe
                df = pd.DataFrame({word_variable: x, measure_name: y, 'predictor': data_type, 'condition': predictors})
                plt.rcParams.update({'font.size': 18})
                if word_variable in ['predictability', 'frequency', 'length', 'word_id']:
                    plot = sb.lmplot(data=df, x=word_variable, y=measure_name, hue='predictor', col='condition')
                else:
                    plot = sb.catplot(data=df, x=word_variable, y=measure_name, hue='predictor', col='condition', kind="violin", split=True)
                # if measure_name in ['first fix duration', 'gaze duration', 'total reading time']:
                #     plot.set(ylim=(50, 400))
                #     plot.set(yticks=[100, 150, 200, 250, 300, 350, 400])
                if measure_name in ['skip', 'single fix', 'regression']:
                    plot.set(ylim=(0, 1.01))
                results_dir = f'{results_dir}/plots'
                if not os.path.isdir(results_dir): os.makedirs(results_dir)
                plot.savefig(f'{results_dir}/plot_pred_trend_{measure_name}_{word_variable}.png')
                plt.close()

def drop_nan_values_4_error(true_values:pd.Series, simulated_values:pd.Series, word_variables:pd.Series):

    values = defaultdict(list)
    counter = 0
    for true_value, simulated_value, word_var in zip(true_values, simulated_values, word_variables):
        if not true_values.isnull().tolist()[counter] and not simulated_values.isnull().tolist()[counter] and not word_variables.isnull().tolist()[counter]:
            values['true'].append(true_value)
            values['pred'].append(simulated_value)
            values['word_var'].append(word_var)
        counter += 1
    return values

def standardize_diff(true_values, simulated_values):

    norm_sim_values = np.divide(np.subtract(simulated_values, np.mean(true_values)), np.std(true_values))
    norm_true_values = np.divide(np.subtract(true_values, np.mean(true_values)), np.std(true_values))
    norm_diff = np.subtract(norm_sim_values, norm_true_values)

    return norm_sim_values, norm_true_values, norm_diff

def compute_root_mean_squared_error(mean_squared_diff):

    # root mean squared error measures the average difference between values predicted by the model
    # and the eye-tracking values. It provides an estimate of how well the model was able to predict the
    # eye-tracking value.
    return math.sqrt(mean_squared_diff)

def compute_error(measure, data_log, word_variable, conditions):

    x, y, predictors = [], [], []
    results_dir = ''

    for predictor in conditions:
        print(predictor)
        print()
        sim_rmse = defaultdict(list)

        for data_name, data in data_log.items():

            if f'simulation_eye_movements_Provo_Corpus_continuous_reading_{predictor}' in data_name:
                results_dir = os.path.dirname(data_name).replace('model_output', 'analysed')
                sim_data = data
            elif 'Provo_Corpus_eye_tracking_last_sim_mean.csv' in data_name or 'Provo_Corpus_eye_tracking_mean.csv' in data_name:
                human_values = data[measure]

        if predictor == 'None': predictor = 'baseline'

        for sim_id, sim_rows in sim_data.groupby('simulation_id'):
            model_values = sim_rows[measure]
            word_values = sim_rows[word_variable]
            # excluding words with nan value, e.g. skipped words of prob 1.
            # print(len(human_values), len(model_values), len(word_values))
            values = drop_nan_values_4_error(human_values, model_values, word_values)
            # print(len(values['pred']))
            assert len(values['pred']) > 0, print(measure, values['pred'])
            assert len(values['pred']) == len(values['true']), print(measure, len(values['pred']), len(values['true']))
            assert len(values['pred']) == len(values['word_var']), print(measure, len(values['pred']), len(values['word_var']))
            # standardize values according to human mean
            norm_sim, norm_true, norm_diff = standardize_diff(values['true'], values['pred'])
            # print(f'Standardized error: {norm_diff}')
            squared_diff = np.square(norm_diff)
            # print(f'Square Error: {squared_diff}')
            assert len(squared_diff) == len(values['word_var']), print(len(squared_diff), len(values['word_var']))
            # print(f'Mean Square Error: {squared_diff.mean()}')
            # print(f'RMSE: {np.sqrt(squared_diff.mean())}')
            # compute rmse for each word variable category/bin
            if word_variable in ['pos_tag', 'pos_cat']:
                for pos_cat in set(values['word_var']):
                    se_per_pos = []
                    for pos_cat_i, se in zip(values['word_var'], squared_diff):
                        if pos_cat_i == pos_cat:
                           se_per_pos.append(se)
                    mean_se_per_pos = np.mean(se_per_pos)
                    y_rmse = np.sqrt(mean_se_per_pos)
                    # sim_rmse[pos_cat].append(y_rmse)
                    x.append(pos_cat)
                    y.append(y_rmse)
                    predictors.append(predictor)
                # print(y)
                # print(x)
                # print(predictors)
            else:
                y_means, edges, _ = binned_statistic(values['word_var'], squared_diff, statistic='mean', bins=20)
                x_bins = edges[:-1]
                # print(f'Mean Square Error for each bin: {y_means}')
                y_rmse = np.sqrt(y_means)
                # print(f'RMSE for each bin: {y_rmse}')
                # print(f'Mean over bins: {np.mean(y_means)}, RMSE over bins: {np.sqrt(np.mean(y_means))}')
                for bin, rmse in zip(x_bins, y_rmse):
                    sim_rmse[bin].append(rmse)
                # predictors.extend([predictor for value in x_bins])
        if sim_rmse:
            for bin, values in sim_rmse.items():
                x.append(bin)
                y.append(np.mean(values))
                predictors.append(predictor)

    return x, y, predictors, results_dir

def plot_rmse_word_variable(measures, data_log, word_variables, conditions):

    for measure in measures:
        print()
        print('-----------------')
        print(measure)
        for word_variable in word_variables:
            print()
            print(word_variable)
            print()
            x, y, predictors, results_dir = compute_error(measure, data_log, word_variable, conditions)

            df = pd.DataFrame({word_variable: x, 'RMSE': y, 'condition': predictors})

            plt.rcParams.update({'font.size': 18})

            if word_variable in ['predictability', 'frequency', 'length', 'word_id']:
                plot = sb.lmplot(data=df, x=word_variable, y='RMSE', hue='condition')
            else:
                plot = sb.catplot(data=df, x=word_variable, y='RMSE', hue='condition', kind='violin', order=['content', 'function', 'other'])

            results_dir = f'{results_dir}/plots'
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)
            plot.savefig(f'{results_dir}/plot_RMSE_trend_simulations_{measure}_{word_variable}.png')
            # plot.figure.savefig(f'{results_dir}/plot_RMSE_trend_simulations_{measure}_{word_variable}.png')
            plt.close()

results_filepaths = ["../data/analysed/_2023_12_05_09-57-49/simulation_eye_movements_Provo_Corpus_continuous_reading_cloze_0.05.csv",
                         "../data/analysed/_2023_12_05_09-57-49/simulation_eye_movements_Provo_Corpus_continuous_reading_gpt2_0.05.csv",
                         "../data/analysed/_2023_12_05_09-57-49/simulation_eye_movements_Provo_Corpus_continuous_reading_llama_0.05.csv",
                         "../data/analysed/_2023_12_05_09-57-49/simulation_eye_movements_Provo_Corpus_continuous_reading_None_0.1.csv",
                         "../data/processed/Provo_Corpus_eye_tracking_mean.csv"]
frequency_filepath = '../data/processed/frequency_map_Provo_Corpus_continuous_reading_english.json'
predictability_filepath = ['../data/processed/prediction_map_Provo_Corpus_cloze_continuous_reading_english.json',
                           '../data/processed/prediction_map_Provo_Corpus_gpt2_continuous_reading_english_topkall_0.01.json',
                           '../data/processed/prediction_map_Provo_Corpus_llama_continuous_reading_english_topkal_0.01.json']
measures = ['skip']

word_factors = ['frequency',
                'length',
                'word_id',
                'pos_cat']

predictors = ['cloze',
              'gpt2',
              'llama']

data_log = dict()
for filepath in results_filepaths:
    data_log[filepath] = pd.read_csv(filepath, sep='\t')

if 'pos_tag' in word_factors:
    nlp = spacy.load("en_core_web_sm")

# add length, frequency, pos tag and position in sentence
data_log_new = dict()
for filepath, df in data_log.items():
    if 'None' in filepath:
        word_factors_base = word_factors.copy()
        word_factors_base.remove('predictability')
        new_df = get_word_factors(word_factors_base, df, frequency_filepath)
    new_df = get_word_factors(word_factors, df, frequency_filepath, predictability_filepath)
    data_log_new[filepath] = new_df
    new_df.to_csv(filepath, sep='\t', index=False)

# data_log_new = dict()
# for filepath in results_filepaths:
#     df = pd.read_csv(filepath, sep='\t')
#     data_log_new[filepath] = df

# # plot results
plot_sim_results(data_log_new, measures, word_factors)

# plot error vs. word factor
plot_rmse_word_variable(measures, data_log_new, word_factors, predictors)