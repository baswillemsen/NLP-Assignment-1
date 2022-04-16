## Scratch file for experimentation and exploration

import pandas as pd
from wordfreq import word_frequency
import matplotlib.pyplot as plt

col_names = ['id','sentence','target_start','target_end','target_words','num_native','num_non_native','difficult_native','difficult_non_native','label_bin','label_prob']
df = pd.read_csv('data/original/english/WikiNews_Dev.tsv', sep='\t', names=col_names)
df.head()

# 7. Extract basic statistics ------------------------------------------------------------------------------------------
label_bin_0 = len(df[df['label_bin']==0])
label_bin_1 = len(df[df['label_bin']==1])
print(label_bin_0, label_bin_1)

label_prob_min = df['label_prob'].min()
label_prob_max = df['label_prob'].max()
label_prob_mean = df['label_prob'].mean()
label_prob_sd = df['label_prob'].std()
print(label_prob_min, label_prob_max, label_prob_mean, label_prob_sd)

len(df[df['target_words'].str.split(" ").apply(len)>1]
df['target_words'].str.split(" ").apply(len).max()

# 8. Explore linguistic characteristics --------------------------------------------------------------------------------
df2 = df[(df['target_words'].str.split(" ").apply(len)==1) & (df['label_bin']!=0)]
len(df2)

def apply_word_freq(token):
    return word_frequency(token,'en')

df2['token_length'] = df2['target_words'].apply(len)
df2['word_freq'] = df2['target_words'].apply(apply_word_freq)

target_words_list = df2['target_words'].to_list()
target_words_str = ' '.join([str(elem) for elem in target_words_list])
len(target_words_str)

doc = nlp(target_words_str)                     ##todo!
pos_tokens = []
for token in doc:
    print(token)
    print(type(token))
    if token != '-':
        pos_tokens.append(token.pos_)
df2['pos_tag'] = pos_tokens
pos_tokens

print(np.corrcoef(df2['token_length'], df2['label_prob'])[0,1])
print(np.corrcoef(df2['word_freq'], df2['label_prob'])[0,1])

plt.scatter(df2['token_length'], df2['label_prob'])
plt.clf()
plt.scatter(df2['word_freq'], df2['label_prob'])
plt.clf()