import chardet
import os
import pandas as pd

# TODO add preprocessing for all experiment stimuli files

filepath = '../data/Provo_Corpus-Predictability_Norms.csv'
encoding = chardet.detect(open(filepath, "rb").read())['encoding']
base_name = os.path.basename(filepath).replace('.txt', '').replace('.csv', '')

if base_name == 'Provo_Corpus-Predictability_Norms':
    data = pd.read_csv(filepath, sep=',', encoding=encoding)
    ids,texts = [],[]
    stim_df = pd.DataFrame()
    for i, text_info in data.groupby(['Text_ID','Text']):
        ids.append(i[0])
        texts.append(i[1])
    stim_df['id'] = ids
    stim_df['all'] = texts
    stim_df.to_csv('../stimuli/Provo_Corpus.csv', sep='\t', index=False)