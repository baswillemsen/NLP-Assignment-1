## Scratch file for experimentation and exploration

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from wordfreq import word_frequency
import matplotlib.pyplot as plt
import spacy
nlp = spacy.load("es_core_news_sm")


col_names = ['id', 'sentence', 'target_start', 'target_end', 'target_words', 'num_native', 'num_non_native',
             'difficult_native', 'difficult_non_native', 'label_bin', 'label_prob']
df = pd.read_csv('data/original/spanish/Spanish_Train.tsv', sep='\t', names=col_names)
df.head()

# 7. Extract basic statistics ------------------------------------------------------------------------------------------
label_bin_0 = len(df[df['label_bin'] == 0])  # Number of instances labeled with 0
label_bin_1 = len(df[df['label_bin'] == 1])  # Number of instances labeled with 1
print(label_bin_0, label_bin_1)

label_prob_min = df['label_prob'].min()  # min
label_prob_max = df['label_prob'].max()  # max
label_prob_median = df['label_prob'].median()  # median
label_prob_mean = df['label_prob'].mean()  # mean
label_prob_sd = df['label_prob'].std()  # standard deviation
print(label_prob_min, label_prob_max, label_prob_median, label_prob_mean, label_prob_sd)

len(df[df['target_words'].str.split(" ").apply(len) > 1])  # Number of instances consisting of more than one token
df['target_words'].str.split(" ").apply(len).max()  # Maximum number of tokens for an instance

# 8. Explore linguistic characteristics --------------------------------------------------------------------------------
df8 = df[(df['target_words'].str.split(" ").apply(len) == 1) & (df['label_bin'] != 0)]  # Filter single tokens, complex by at least 1 annotator, add to df8
len(df8)  # number of tokens


def apply_word_freq(token):
    return word_frequency(token, 'es')


df8['token_length'] = df8['target_words'].apply(len)  # make a column with the character length of the target word
df8['word_freq'] = df8['target_words'].apply(apply_word_freq)  # make a column with the word freq of the target word

target_words_list = df8['target_words'].to_list()
target_words_str = ' '.join([str(elem) for elem in target_words_list])
target_words_str = target_words_str.replace('-', '')  # delete all the '-'s, join words
target_words_str = target_words_str.replace('.', '')  # delete all the '-'s, join words
target_words_str = target_words_str.replace('/', '')  # delete all the '-'s, join words
target_words_str = target_words_str.replace('”', "")  # delete all the '-'s, join words
target_words_str = target_words_str.replace('’', "")  # delete all the '-'s, join words
target_words_str = target_words_str.replace('`', "")  # delete all the '-'s, join words
target_words_str = target_words_str.replace('\'', "")  # delete all the '-'s, join words
target_words_str = target_words_str.replace('»', "")  # delete all the '-'s, join words
target_words_str = target_words_str.replace('«', "")  # delete all the '-'s, join words
target_words_str = target_words_str.replace('مأرب', "")  # delete all the '-'s, join words
target_words_str = target_words_str.replace('UD', "")  # delete all the '-'s, join words
target_words_str = target_words_str.replace('NE', "")  # delete all the '-'s, join words
target_words_str = target_words_str.replace('FC', "")  # delete all the '-'s, join words
target_words_str = target_words_str.replace('\xa0a', "")  # delete all the '-'s, join words
target_words_str = target_words_str.replace('', "")  # delete all the '-'s, join words
target_words_str = target_words_str.replace('I\xa0a.', "")  # delete all the '-'s, join words
target_words_str = target_words_str.replace('\xa0a.\xa0C a.\xa0C', "")  # delete all the '-'s, join words
target_words_str

doc = nlp(target_words_str)
pos_text = []
pos_tokens = []
for token in doc:
    print(token.text)
    pos_text.append(token.text)
    pos_tokens.append(token.pos_)
df8['pos_tag'] = pos_tokens
print(df8)

print(np.corrcoef(df8['token_length'], df8['label_prob'])[0, 1])  # corr token length and perceived complexity word
print(np.corrcoef(df8['word_freq'], df8['label_prob'])[0, 1])  # corr word frequency and perceived complexity word

plt.scatter(df8['token_length'], df8['label_prob'])
plt.clf
plt.scatter(df8['word_freq'], df8['label_prob'])
plt.clf()
plt.scatter(df8['word_freq'], df8['pos_tag'])
plt.clf()
