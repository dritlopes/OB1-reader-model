import chardet
import os
import pandas as pd

# TODO add preprocessing for all experiment stimuli files

filepath = '../data/predictability/Provo_Corpus-Predictability_Norms.csv'
# encoding = chardet.detect(open(filepath, "rb").read())['encoding']
base_name = os.path.basename(filepath).replace('.txt', '').replace('.csv', '')

if base_name == 'Provo_Corpus-Predictability_Norms':

    data = pd.read_csv(filepath, sep=',', encoding="ISO-8859-1")
    ids, texts, words, word_ids = [],[],[],[]
    for i, text_info in data.groupby('Text_ID'):
        ids.append(int(i))
        texts.append(text_info['Text'].tolist()[0])
        words.append(text_info['Word'].unique().tolist())
        word_ids.append(text_info['Word_Number'].unique().tolist())
    stim_df = pd.DataFrame(data={'id': ids,
                                'all': texts,
                                'words': words,
                                'word_ids': word_ids})
    stim_df.to_csv('../stimuli/Provo_Corpus.csv', sep='\t', index=False)

