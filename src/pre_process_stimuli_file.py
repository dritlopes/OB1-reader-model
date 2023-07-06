import chardet
import os
import pandas as pd
from utils import pre_process_string

# TODO add preprocessing for all experiment stimuli files

filepath = '../data/predictability/Provo_Corpus-Predictability_Norms.csv'
# encoding = chardet.detect(open(filepath, "rb").read())['encoding']
base_name = os.path.basename(filepath).replace('.txt', '').replace('.csv', '')

# pre_process stimuli file
if base_name == 'Provo_Corpus-Predictability_Norms':
    data = pd.read_csv(filepath, sep=',', encoding="ISO-8859-1")
    ids, texts, words, word_ids = [],[],[],[]
    for i, text_info in data.groupby('Text_ID'):
        ids.append(int(i)-1)
        text = text_info['Text'].tolist()[0]
        # fix error in text 36 in raw data
        if int(i) == 36:
            text = text.replace(' Ñ','')
        texts.append(text)
        text_words = [pre_process_string(token) for token in text.split()]
        text_word_ids = [i for i in range(0,len(text_words))]
        words.append(text_words)
        word_ids.append(text_word_ids)
        # items = []
        # for word, word_id in zip(text_info['Word'].tolist(),text_info['Word_Number'].tolist()):
        #     if (word, word_id) not in items: items.append((word,word_id))
        # words.append([word for word, word_id in items])
        # word_ids.append([word_id for word, word_id in items])
    stim_df = pd.DataFrame(data={'id': ids,
                                'all': texts,
                                'words': words,
                                'word_ids': word_ids})
    stim_df.to_csv('../stimuli/Provo_Corpus.csv', sep='\t', index=False)






