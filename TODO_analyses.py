# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt
import spacy
from collections import Counter
import pandas as pd
# from nltk import word_tokenize
# from nltk.util import ngrams

nlp = spacy.load('en_core_web_sm')
input = open("sentences.txt", encoding="utf8")
lines = input.read()
input.close()
doc = nlp(lines)

## PART A
# 1. Tokanization
# word_frequencies = Counter()
# num_sentences = 0
# words_length = 0
# for sentence in doc.sents:
#     words = []
#     num_sentences += 1
#     for token in sentence:
#         # Let's filter out punctuation
#         if not token.is_punct:
#             words.append(token.text)
#             words_length = words_length + len(token)
#     word_frequencies.update(words)
#
# num_tokens = len(doc)
# num_types = len(word_frequencies.keys())
# num_words = sum(word_frequencies.values())
# avg_words_per_sentence = sum(word_frequencies.values()) / num_sentences
# avg_words_length = words_length / num_words
#
# print(num_tokens, num_words, num_types, avg_words_per_sentence, avg_words_length)

# # 2. 2.	Word Classes
# POStoken = []
# POSpos = []
# POStags = []
# for token in doc:
#     if not token.is_punct:
#         POStoken.append(token.text)
#         POSpos.append(token.pos_)
#         POStags.append(token.tag_)
# df = pd.DataFrame({"POStoken": POStoken,"POSpos": POSpos,"POStags": POStags})
#
# print(df['POStags'].value_counts())
# print(df['POStags'].value_counts().sum())
#
# print(df[df['POStags'] == 'DT']['POStoken'].value_counts())

# # 3.	N-grams
# BIGtoken = []
# BIGtags = []
# for i in range(len(doc)-1):
#     token = [doc[i], doc[i+1]]
#     if not token.is_punct:
#         BIGtoken.append(token.text)
#         BIGtags.append(token.tag_)
# df2 = pd.DataFrame({"BIGtoken": BIGtoken,"BIGtags": BIGtags})
#
# print(df2['BIGtags'])

# # 4.	Lemmatization
# LEMtoken = []
# LEMpos = []
# LEMlem = []
# for token in doc:
#     if not token.is_punct:
#         LEMtoken.append(token.text)
#         LEMpos.append(token.pos_)
#         LEMlem.append(token.lemma_)
# df3 = pd.DataFrame({"LEMtoken": LEMtoken,"LEMpos": LEMpos,"LEMlem": LEMlem})
#
# print(df3[df3['LEMpos'] == 'VERB']['LEMlem'].value_counts())
# print(df3[df3['LEMlem'] == 'find'])

# 5.	NER
NERtoken = []
NERent = []
for ent in doc.ents:
    NERtoken.append(ent.text)
    NERent.append(ent.label_)
df5 = pd.DataFrame({"NERtoken": NERtoken,"NERent": NERent})

print(df5.count())
print(df5['NERent'].unique())

