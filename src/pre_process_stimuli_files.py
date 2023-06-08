import chardet
import os
import pandas as pd

# TODO add preprocessing for all experiment stimuli files

filepath = '../data/predictability/Provo_Corpus-Predictability_Norms.csv'
# encoding = chardet.detect(open(filepath, "rb").read())['encoding']
base_name = os.path.basename(filepath).replace('.txt', '').replace('.csv', '')

if base_name == 'Provo_Corpus-Predictability_Norms':

    data = pd.read_csv(filepath, sep=',', encoding="ISO-8859-1")
    ids,texts = [],[]
    stim_df = pd.DataFrame()
    for i, text_info in data.groupby(['Text_ID']):
        ids.append(int(i))
        texts.append(text_info['Text'].tolist()[0])
    stim_df['id'] = ids
    stim_df['all'] = texts
    stim_df.to_csv('../stimuli/Provo_Corpus.csv', sep='\t', index=False)

