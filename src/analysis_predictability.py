import pandas as pd
import pickle
import json
import seaborn as sb
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, LlamaTokenizer
from collections import Counter

def read_in_pred_files(pred_map_filepaths):

    pos_pred_maps = dict()

    for predictor, filepath in pred_map_filepaths.items():
        with open(filepath) as f:
            pos_pred_maps[predictor]=json.load(f)

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

def text_words_predictions(pos_pred_map, predictor):

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

def compute_frequency_predictions(predictions):

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

def compute_unknown_proportion(pred_maps, pred_maps_unknown):

    predictors, proportion = [],[]

    for predictor in pred_maps.keys():
        proportions = []
        for text, info in pred_maps_unknown[predictor].items():
            for pos, pos_info in info.items():
                # change this later when cloze is updated (also having token_processed key)
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

    assert len(predictors) == len(proportion)

    return predictors, proportion

def find_multi_token_targets(pred_map, predictor):

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


# read in predictions
pred_map_filepaths = {'cloze':'../data/processed/prediction_map_Provo_Corpus_cloze_continuous_reading_english.json',
                      'GPT2': '../data/processed/prediction_map_Provo_Corpus_gpt2_continuous_reading_english_topkall.json',
                      'LLAMA': '../data/processed/prediction_map_Provo_Corpus_llama_continuous_reading_english_topkall.json'}
pred_maps = read_in_pred_files(pred_map_filepaths)

target_predictions = defaultdict(list)
for predictor, data in pred_maps.items():
    pred_values = text_words_predictions(data,predictor)
    target_predictions['text_word'].extend(pred_values['text_word'])
    target_predictions['prediction'].extend(pred_values['prediction'])
    target_predictions['predictor'].extend(pred_values['predictor'])

# write out mappings in csv to visualise it more easily
for predictor, pred_map in pred_maps.items():
    df = convert_json_to_csv(pred_map)
    if predictor == 'cloze': df = df.sort_values(by=['text_id','word_id'])
    filepath = pred_map_filepaths[predictor].replace('.json','.csv')
    df.to_csv(filepath, sep='\t', index=False)

# count pred values in each predictor only including the predictability of target words (words in text)
pred_value_counts_col = compute_frequency_predictions(target_predictions)
pred_value_counts_df = pd.DataFrame.from_dict(pred_value_counts_col)
pred_value_counts_df.sort_values(by='predictability', ascending=True, inplace=True)
pred_value_counts_df.to_csv('../data/processed/predictions_distribution.csv',sep='\t', index=False)
# sb.relplot(data=pred_value_counts_col, x = 'predictability', y = 'proportion', hue = 'predictor', kind = 'scatter')

# proportion of unknown predicted tokens per target word in relation to model lexicon
unknown_map_filepaths = {'cloze': '../data/processed/prediction_map_Provo_Corpus_cloze_continuous_reading_english_unknown.json',
                         'GPT2': '../data/processed/prediction_map_Provo_Corpus_gpt2_continuous_reading_english_topkall_unknown.json',
                         'LLAMA': '../data/processed/prediction_map_Provo_Corpus_llama_continuous_reading_english_topkall_unknown.json'}

pred_maps_unknown = read_in_pred_files(unknown_map_filepaths)
predictors,proportion = compute_unknown_proportion(pred_maps,pred_maps_unknown)
# sb.set(rc={'figure.figsize': (10,5)})
# ax = sb.barplot(x=predictors, y=proportion)
# for i in ax.containers:
#     ax.bar_label(i,)
pd.DataFrame({'predictor': predictors, "proportion": proportion}).to_csv(f'../data/processed/propotion_unknown_tokens.csv',sep='\t',index=False)

# analyse unknown tokens predicted by language model
for predictor in ['GPT2', 'LLAMA']:
    pred_tokens = []
    for text, info in pred_maps_unknown[predictor].items():
        for pos, pos_info in info.items():
            pred_tokens.extend(pos_info['predictions'].keys())
    unknown_tokens = Counter(pred_tokens)

    df_unknown = pd.DataFrame({'TOKEN': [token for token in unknown_tokens.keys()], 'COUNT': [count for count in unknown_tokens.values()]})
    df_unknown = df_unknown.sort_values(by='COUNT',ascending=False)
    df_unknown.to_csv(f'../data/processed/unknown_counts_{predictor}.csv',sep='\t', index=False)

    df_unknown_short = pd.DataFrame({'TOKEN': [token for token in unknown_tokens.keys() if len(token) <= 2], 'COUNT': [count for token, count in unknown_tokens.items() if len(token) <= 2]})
    df_unknown_short = df_unknown_short.sort_values(by='COUNT',ascending=False)
    df_unknown_short.to_csv(f'../data/processed/unknown_short_counts_{predictor}.csv',sep='\t',index=False)

# check the average number of predictions from language model
for predictor in ['GPT2', 'LLAMA']:
    n_predictions = []
    for text, info in pred_maps[predictor].items():
        for pos, pos_info in info.items():
            predictions = pos_info['predictions'].keys()
            n_predictions.append(len(predictions))
    print(f'Average number of predictions {predictor}: ', round(sum(n_predictions)/len(n_predictions)))
    print('Range: ', min(n_predictions), '-',max(n_predictions))

# check which text words correspond to a multi-token in language model
for predictor, pred_map in pred_maps.items():
    find_multi_token_targets(pred_map, predictor)

# measure word prediction accuracy
for predictor, texts in pred_maps.items():
    acc = []
    for info in texts.values():
        for predictions in info.values():
            target = predictions['target']
            if target in predictions['predictions'].keys():
                acc.append(1)
            else:
                acc.append(0)
    acc_score = sum(acc)/len(acc)
    print(f'{predictor} accuracy: {acc_score}')

# inspect lexicon
with open('../data/processed/lexicon.pkl', 'rb') as infile:
      lexicon = pickle.load(infile)
print('Lexicon:', lexicon)