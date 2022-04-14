# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt
import spacy
import pandas as pd
import nltk
from collections import Counter
from nltk import word_tokenize
from nltk.util import ngrams
import lemminflect

nlp = spacy.load('en_core_web_sm')
input = open("sentences.txt", encoding="utf8")
lines = input.read()
input.close()
doc = nlp(lines)

# # PART A
# 1. Tokanization
token_counter = Counter()
num_sentences = 0
num_words = 0
words_length = 0
for sentence in doc.sents:
    words = []
    num_sentences += 1
    for token in sentence:
        words.append(token.text)
        if not token.is_punct:
            num_words += 1
            words_length += len(token)
    token_counter.update(words)

num_tokens = sum(token_counter.values())
num_types = len(token_counter.keys())
num_words = num_words
num_sentences = num_sentences
avg_words_per_sentence = num_words / num_sentences
avg_words_length = words_length / num_words

print(num_tokens, num_words, num_types, avg_words_per_sentence, avg_words_length)

# 2. Word Classes
WORDtoken = []
WORDpos = []
WORDtag = []
for token in doc:
    WORDtoken.append(token.text)
    WORDpos.append(token.pos_)
    WORDtag.append(token.tag_)
df2 = pd.DataFrame({"WORDtoken": WORDtoken,"WORDpos": WORDpos,"WORDtag": WORDtag})

print(df2['WORDpos'].value_counts().head(10)) #number of tags per POS tag
print(df2['WORDpos'].value_counts().sum()) #total number of POS tags

top10POStags = ['NOUN','PROPN','PUNCT','VERB','ADP','DET','ADJ','AUX','PRON','SPACE']
for tag in top10POStags:
    print(tag, ': \n')
    print(df2[df2['WORDpos'] == tag]['WORDtag'].value_counts())
    print(df2[df2['WORDpos'] == tag]['WORDtoken'].value_counts().head(3))
    print(df2[df2['WORDpos'] == tag]['WORDtoken'].value_counts().tail(1))

# 3. N-grams
doc_str = str(doc)
all_sentences = doc_str.split("\n")
print(all_sentences)

for n in [2,3]:
    bigrams_counter = Counter()
    bigrams = []
    for line in all_sentences:
        token = nltk.word_tokenize(line)
        bigram = list(ngrams(token, n))
        for gram in bigram:
            bigrams.append(gram)

    bigrams_counter.update(bigrams)
    bigramstop3 = bigrams_counter.most_common(3)
    print(bigramstop3)
    for bigram in bigramstop3:
        for word in (bigram[0]):
            print(nltk.pos_tag([word]))

# 4. Lemmatization
lem_counter = Counter()
for token in doc:
    if not token.is_punct:
        LEMtoken.append(token.text)
        LEMpos.append(token.pos_)
        LEMlem.append(token.lemma_)
df3 = pd.DataFrame({"LEMtoken": LEMtoken,"LEMpos": LEMpos,"LEMlem": LEMlem})

print(df3[df3['LEMpos'] == 'VERB']['LEMlem'].value_counts())
print(df3[df3['LEMlem'] == 'find'])

# 5. NER
NERtoken = []
NERent = []
for ent in doc.ents:
    NERtoken.append(ent.text)
    NERent.append(ent.label_)
df5 = pd.DataFrame({"NERtoken": NERtoken,"NERent": NERent})

print(df5.count())
print(df5['NERent'].unique())
print(df5.head(10))

