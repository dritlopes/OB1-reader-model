import pandas as pd
import pickle
import json
import seaborn as sb
from scipy.stats import binned_statistic
from collections import defaultdict
import sys
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, LlamaTokenizer
from collections import Counter
import os
import scipy.stats as stats
import itertools

def read_in_pred_files(pred_map_filepaths):

    pos_pred_maps = dict()

    for predictor, filepath in pred_map_filepaths.items():
        with open(filepath) as f:
            pos_pred_maps[predictor] = json.load(f)

    return pos_pred_maps

def convert_json_to_csv(pred_map):

    text_id, word_id, target, prediction, value = [],[],[],[],[]
    for text, text_info in pred_map.items():
        for pos, pos_info in text_info.items():
            for pred, pred_value in pos_info['predictions'].items():
                text_id.append(int(text))
                word_id.append(int(pos))
                target.append(pos_info['target'])
                prediction.append(pred)
                value.append(pred_value)
    df = pd.DataFrame({'text_id': text_id,
                       'word_id': word_id,
                       'target': target,
                       "prediction": prediction,
                       "predictability": value})
    return df

def write_out_mappings_csv(pred_maps, pred_map_filepaths):

    for predictor, pred_map in pred_maps.items():
        df = convert_json_to_csv(pred_map)
        if predictor == 'cloze': df = df.sort_values(by=['text_id', 'word_id'])
        filepath = pred_map_filepaths[predictor].replace('.json', '.csv')
        df.to_csv(filepath, sep='\t', index=False)

def text_words_predictions(pos_pred_map, predictor):

    """ Find the predictability value of each text word by each predictor"""

    pred_values = defaultdict(list)

    for text, words in pos_pred_map.items():
        for word, predictions in words.items():
            pred_value = 0.0
            target = predictions['target']
            if target in predictions['predictions'].keys():
                pred_value = predictions['predictions'][target]
            if pred_value <= 1:  # to exclude mistaken value from cloze data: 1.025
                pred_values['text_word'].append(target)
                pred_values['prediction'].append(pred_value)
                pred_values['predictor'].append(predictor)
    print(f'{predictor} proportion target words not predicted: {pred_values["prediction"].count(0.0)} from {len(pred_values["text_word"])}')

    return pred_values

def get_text_word_pred(pred_maps):

    target_predictions = defaultdict(list)
    for predictor, data in pred_maps.items():
        pred_values = text_words_predictions(data, predictor)
        target_predictions['text_word'].extend(pred_values['text_word'])
        target_predictions['prediction'].extend(pred_values['prediction'])
        target_predictions['predictor'].extend(pred_values['predictor'])
    return target_predictions

def compute_frequency_predictions(predictions):

    """ Find how frequent each predictability value is for each predictor"""

    pred_values, counts, predictors, proportion = [],[],[],[]

    for predictor in set(predictions['predictor']):
        pred_value_counts = defaultdict(int)
        for prediction, a_predictor in zip(predictions['prediction'], predictions['predictor']):
            if predictor == a_predictor:
                pred_value_counts[prediction] += 1
        pred_values.extend(pred_value_counts.keys())
        counts.extend(pred_value_counts.values())
        proportion.extend([value/sum(pred_value_counts.values()) for value in pred_value_counts.values()])

        for pred_value in range(len(pred_value_counts.keys())):
            predictors.append(predictor)

        assert len(pred_values) == len(counts) == len(predictors), print(len(pred_values),len(counts),len(predictors))

    pred_value_counts_col = {'predictability': pred_values,
                             'counts': counts,
                             'proportion': proportion,
                             'predictor': predictors}
    return pred_value_counts_col

def count_text_word_pred(target_predictions):

    pred_value_counts_col = compute_frequency_predictions(target_predictions)
    pred_value_counts_df = pd.DataFrame.from_dict(pred_value_counts_col)
    pred_value_counts_df.sort_values(by='predictability', ascending=True, inplace=True)
    pred_value_counts_df.to_csv('../data/processed/predictions_distribution.csv',sep='\t', index=False)
    plot = sb.relplot(data=pred_value_counts_col, x = 'predictability', y = 'proportion', hue = 'predictor', kind = 'scatter')
    plot.figure.savefig('../data/processed/predictions_distribution.jpg')
    plt.close()

def compute_unknown_proportion(pred_maps, pred_maps_unknown):

    """Find the proportion of predicted tokens/words which are not in the lexicon of OB1-reader"""

    predictors, proportion = [],[]

    for predictor in pred_maps.keys():
        proportions = []
        for text, info in pred_maps_unknown[predictor].items():
            for pos, pos_info in info.items():
                if predictor == 'cloze':
                    unknown_predictions = pos_info['predictions'].keys()
                else:
                    unknown_predictions = set([pred_token['token_processed'] for pred_token in pos_info['predictions'].values()])
                all_predictions = pred_maps[predictor][text][pos]['predictions'].keys()
                if len(all_predictions) > 0:
                    proportion_unknown = len(unknown_predictions)/len(all_predictions)
                    proportions.append(proportion_unknown)

        proportion.append(sum(proportions)/len(proportions))
        predictors.append(predictor)
        print(f'Proportion of predicted tokens by {predictor} not in OB1-reader lexicon: {sum(proportions)/len(proportions)}')

    assert len(predictors) == len(proportion)

    # sb.set(rc={'figure.figsize': (10,5)})
    # ax = sb.barplot(x=predictors, y=proportion)
    # for i in ax.containers:
    #     ax.bar_label(i,)
    # ax.figure.savefig('../data/processed/proportion_unknown_predictions.jpg')
    # plt.close()
    pd.DataFrame({'predictor': predictors, "proportion": proportion}).to_csv(f'../data/processed/propotion_unknown_tokens.csv',sep='\t',index=False)

def analyse_unk_word_pred(pred_maps_unknown):

    for predictor in ['GPT2', 'LLAMA']:
        if predictor in pred_maps_unknown.keys():
            pred_tokens = []
            for text, info in pred_maps_unknown[predictor].items():
                for pos, pos_info in info.items():
                    pred_tokens.extend(pos_info['predictions'].keys())
            unknown_tokens = Counter(pred_tokens)
            df_unknown = pd.DataFrame({'TOKEN': [token for token in unknown_tokens.keys()],
                                       'COUNT': [count for count in unknown_tokens.values()]})
            df_unknown = df_unknown.sort_values(by='COUNT', ascending=False)
            df_unknown.to_csv(f'../data/processed/unknown_counts_{predictor}.csv', sep='\t', index=False)
            df_unknown_short = pd.DataFrame({'TOKEN': [token for token in unknown_tokens.keys() if len(token) <= 2],
                                             'COUNT': [count for token, count in unknown_tokens.items() if
                                                       len(token) <= 2]})
            df_unknown_short = df_unknown_short.sort_values(by='COUNT', ascending=False)
            df_unknown_short.to_csv(f'../data/processed/unknown_short_counts_{predictor}.csv', sep='\t', index=False)

def get_mean_count_pred(pred_maps):

    for predictor in ['GPT2', 'LLAMA']:
        n_predictions = []
        for text, info in pred_maps[predictor].items():
            for pos, pos_info in info.items():
                predictions = pos_info['predictions'].keys()
                n_predictions.append(len(predictions))
        print(f'Average number of predictions {predictor}: ', round(sum(n_predictions)/len(n_predictions)))
        print('Range: ', min(n_predictions), '-',max(n_predictions))

def find_multi_token_targets(pred_maps):

    """ Find which text words are tokenized as multi-tokens by language model tokenizer"""

    for predictor, pred_map in pred_maps.items():

        counts = {'multi': 0,
                  'total': 0}
        targets, tokens, preds, acc = [], [], [],[]
        tokenizer = None

        if predictor == 'GPT2':
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        elif predictor == 'LLAMA':
            tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf", legacy=False)

        if tokenizer:
            for text, info in pred_map.items():
                for pos, pos_info in info.items():
                    target_word = pos_info['target']
                    target_ids = tokenizer.encode(target_word, return_tensors='pt')
                    target_tokens = [tokenizer.decode(token) for token in target_ids[0]]
                    # deals with strange quirk from llama, which tokenizes words sometimes with an unknown token as first token
                    target_tokens = [token for token in target_tokens if token != '<unk>']
                    if len(target_tokens) > 1:
                        counts['multi'] += 1
                        targets.append(target_word)
                        tokens.append(target_tokens)
                        preds.append(list(pos_info['predictions'].keys()))
                        if target_word in pos_info['predictions'].keys():
                            acc.append(1)
                        else:
                            acc.append(0)
                    counts['total'] += 1

            print(f'Multi-token target words from {predictor}: ', round((counts["multi"]/counts["total"])*100), '%')
            multi_token_df = pd.DataFrame({'target':targets, 'tokens':tokens, 'predictions':preds, 'accuracy':acc})
            multi_token_df.to_csv(f'../data/processed/target_multi_tokens_{predictor}.csv',sep='\t',index=False)

def word_pred_acc(pred_maps):

    for predictor, texts in pred_maps.items():
        text_ids, target_ids, targets, preds, accuracy = [], [], [], [], []
        for text_id, info in texts.items():
            for word_id, predictions in info.items():
                target = predictions['target']
                acc = 0
                if target in predictions['predictions'].keys():
                    acc = 1
                text_ids.append(int(text_id))
                target_ids.append(int(word_id))
                targets.append(target)
                preds.append(list(predictions['predictions'].keys()))
                accuracy.append(acc)
        acc_score = sum(accuracy)/len(accuracy)
        print(f'{predictor} accuracy: {acc_score}')
        df = pd.DataFrame({'text_id': text_ids,
                           'word_id': target_ids,
                           'word': targets,
                           'predictions': preds,
                           'accuracy': accuracy})
        if predictor == 'cloze': df = df.sort_values(by=['text_id', 'word_id'])
        df.to_csv(f'../data/processed/prediction_accuracy_{predictor}.csv', sep='\t', index=False)

def drop_nan_values(list1, list2):

    list1_updated, list2_updated = [], []
    list1_nan = np.isnan(list1).tolist()
    list2_nan = np.isnan(list2).tolist()

    assert len(list1) == len(list2), print(len(list1), len(list2))

    for i in range(len(list1)):
        if not list1_nan[i] and not list2_nan[i]:
            list1_updated.append(list1[i])
            list2_updated.append(list2[i])

    return list1_updated, list2_updated

def plot_sim_results_pred(filepaths, measures):

    data_log = dict()
    for filepath in filepaths:
        data_log[filepath] = pd.read_csv(filepath, sep='\t')

    # predictability in x-axis and eye movement measure in y-axis
    for measure in measures:

        x, y, data_type, predictors = [], [], [], []

        for predictor in ['cloze', 'gpt2', 'llama']:

            model_values, human_values, pred_values = [], [], []

            for data_name, data in data_log.items():

                if f'simulation_eye_movements_mean_Provo_Corpus_continuous_reading_{predictor}' in data_name:
                    results_dir = os.path.dirname(data_name).replace('model_output', 'analysed')
                    pred_values = data['predictability'].tolist()
                    model_values = data[measure].tolist()
                elif 'Provo_Corpus_eye_tracking_last_sim_mean.csv' in data_name or 'Provo_Corpus_eye_tracking_mean.csv' in data_name:
                    human_values = data[measure].tolist()

            if len(model_values) > 0 and len(human_values) > 0 and len(pred_values) > 0:
                for eye_movement in [model_values, human_values]:
                    # drop nan values
                    pred_values_clean, eye_movement_clean = drop_nan_values(np.array(pred_values), np.array(eye_movement))
                    # bin data
                    y_means, edges, _ = binned_statistic(pred_values_clean, eye_movement_clean, statistic='mean', bins=50)
                    x_bins = edges[:-1]
                    x.extend(x_bins)
                    y.extend(y_means)
                    if eye_movement == model_values:
                        type = 'OB1-reader'
                    else:
                        type = 'PROVO'
                    data_type.extend([type for bin in x_bins])
                    predictors.extend([predictor for bin in x_bins])

        if len(data_type) > 0:
            # seaborn lmplot function requires dataframe
            df = pd.DataFrame({'predictability': x, measure: y, 'predictor': data_type, 'condition': predictors})

            plot = sb.lmplot(data=df, x='predictability', y=measure, hue='predictor', col='condition')
            if measure in ['first_fix_duration', 'gaze_duration', 'total_reading_time']:
                plot.set(ylim=(100, 350))
            elif measure in ['skip', 'single_fix', 'regression']:
                plot.set(ylim=(0, 1.01))
            results_dir = f'{results_dir}/plots'
            if not os.path.isdir(results_dir): os.makedirs(results_dir)
            plot.figure.savefig(f'{results_dir}/plot_pred_trend_{measure}.png')
            plt.close()

            plot = sb.relplot(data=df, x='predictability', y=measure, hue='predictor', col='condition', kind="line")
            if measure in ['first_fix_duration', 'gaze_duration', 'total_reading_time']:
                plot.set(ylim=(100, 350))
            elif measure in ['skip', 'single_fix', 'regression']:
                plot.set(ylim=(0, 1.01))
            plot.figure.savefig(f'{results_dir}/plot_pred_line_{measure}.png')
            plt.close()

def test_correlation(pred_values, eye_movement_values, filepath):

    # do pearson correlation test
    test = stats.pearsonr(pred_values, eye_movement_values)
    corr_results = {'test': ['corr-coefficient', 'p-value', 'degrees-of-freedom'],
                    'result': [test.statistic, test.pvalue, len(pred_values)-2]}

    df = pd.DataFrame.from_dict(corr_results)
    df.to_csv(filepath, sep='\t', index=False)

def test_correlation_pred(eye_movement_filepath, measures, pred_maps):

    eye_movement_data = pd.read_csv(eye_movement_filepath, sep='\t')
    all_pred_values = dict()

    for predictor, data in pred_maps.items():
        pred_values = []
        for id, row in eye_movement_data.groupby(['text_id', 'word_id', 'word']):
            predictability = 0.0
            # if text in pred map
            if str(id[0]) in data.keys():
                # if word position in pred map
                if str(id[1]) in data[str(id[0])].keys():
                    # if text word among predictions
                    if id[2] in data[str(id[0])][str(id[1])]['predictions'].keys():
                        predictability = data[str(id[0])][str(id[1])]['predictions'][id[2]]
            pred_values.append(predictability)
        for measure in measures:
            filepath = f'../data/processed/pearson_corr_{measure}_{predictor}.csv'
            test_correlation(pred_values, eye_movement_data[measure].tolist(), filepath)
        all_pred_values[predictor] = pred_values

    # test correlation between predictability conditions
    for combi in itertools.combinations(list(all_pred_values.keys()), 2):
        filepath = f'../data/processed/pearson_corr_{combi[0]}_{combi[1]}.csv'
        test_correlation(all_pred_values[combi[0]], all_pred_values[combi[1]], filepath)


def plot_pred_dist(predictions, ):

    df = pd.DataFrame({'predictor': predictions['predictor'],
                       'prediction': predictions['prediction'],
                       'text_word': predictions['text_word']})
    plot = sb.displot(df, x="prediction", stat='probability', col='predictor', common_norm=False)
    plot.figure.savefig('../data/processed/distribution_pred_values.jpg')

def main():

    pred_map_filepaths = {'cloze':'../data/processed/prediction_map_Provo_Corpus_cloze_continuous_reading_english.json',
                          'GPT2': '../data/processed/prediction_map_Provo_Corpus_gpt2_continuous_reading_english_topkall.json',
                          'LLAMA': '../data/processed/prediction_map_Provo_Corpus_llama_continuous_reading_english_topkall.json'}
    unknown_map_filepaths = {'cloze': '../data/processed/prediction_map_Provo_Corpus_cloze_continuous_reading_english_unknown.json',
                             'GPT2': '../data/processed/prediction_map_Provo_Corpus_gpt2_continuous_reading_english_topkall_unknown.json',
                             'LLAMA': '../data/processed/prediction_map_Provo_Corpus_llama_continuous_reading_english_topkall_unknown.json'}
    results_filepaths = ["../data/analysed/_31_10_2023_09-39-35/simulation_eye_movements_mean_Provo_Corpus_continuous_reading_cloze_0.2.csv",
                        "../data/analysed/_31_10_2023_09-39-35/simulation_eye_movements_mean_Provo_Corpus_continuous_reading_llama_0.2.csv",
                         "../data/processed/Provo_Corpus_eye_tracking_mean.csv"]
    eye_movement_filepath = '../data/processed/Provo_Corpus_eye_tracking_mean.csv'
    measures = ['skip',
                'single_fix',
                'first_fix_duration',
                'gaze_duration',
                'total_reading_time',
                'regression_in']

    # read in predictions
    pred_maps = read_in_pred_files(pred_map_filepaths)

    # # write out mappings in csv to visualise it more easily
    # write_out_mappings_csv(pred_maps, pred_map_filepaths)
    #
    # get text word predictability values
    target_predictions = get_text_word_pred(pred_maps)

    # # count pred values in each predictor only including the predictability of target words (words in text)
    # count_text_word_pred(target_predictions)

    # plot distribution of predictability values of text words
    plot_pred_dist(target_predictions)

    # plot distribution of predictability values of all predicted words

    # # proportion of unknown predicted tokens per target word in relation to model lexicon
    # pred_maps_unknown = read_in_pred_files(unknown_map_filepaths)
    # compute_unknown_proportion(pred_maps, pred_maps_unknown)
    #
    # # analyse unknown tokens predicted by language model
    # analyse_unk_word_pred(pred_maps_unknown)
    #
    # # check the average number of predictions from language model
    # get_mean_count_pred(pred_maps)
    #
    # # check which text words correspond to a multi-token in language model
    # find_multi_token_targets(pred_maps)
    #
    # # measure word prediction accuracy
    # word_pred_acc(pred_maps)

    # plot results on predictability
    # plot_sim_results_pred(results_filepaths, measures)

    # test correlation between predictability and eye movements
    # test_correlation_pred(eye_movement_filepath, measures, pred_maps)

if __name__ == '__main__':
    main()
