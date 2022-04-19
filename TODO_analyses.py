# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt
import spacy
import pandas as pd
import nltk
from collections import Counter
from nltk import word_tokenize
from nltk.util import ngrams

nlp = spacy.load('en_core_web_sm')
file_path = 'data/preprocessed/train/sentences.txt'
lines = open(file_path, encoding='utf-8').read()
doc = nlp(lines)

# # PART A
# 1. Tokanization ------------------------------------------------------------------------------------------------------
token_counter = Counter()
num_sentences = 0
num_words = 0
words_length = 0
for sentence in doc.sents:
    words = []
    num_sentences += 1 #count number of sentences
    for token in sentence:
        words.append(token.text) #get all words, incl punctuation as tokens
        if not token.is_punct and token.text != '\n': #exclude punctuation and linebreaks as words
            num_words += 1 #count number of words
            words_length += len(token) #count total word lengths
    token_counter.update(words)

num_tokens = sum(token_counter.values())
num_words = num_words
num_types = len(token_counter.keys())
avg_words_per_sentence = num_words / num_sentences
avg_words_length = words_length / num_words

print(num_tokens, num_words, num_types, avg_words_per_sentence, avg_words_length)

# 2. Word Classes ------------------------------------------------------------------------------------------------------
WORDtoken = []
WORDpos = []
WORDtag = []
for token in doc:
    WORDtoken.append(token.text)
    WORDpos.append(token.pos_)
    WORDtag.append(token.tag_)
df2 = pd.DataFrame({"WORDtoken": WORDtoken,"WORDpos": WORDpos,"WORDtag": WORDtag}) #all results in df2

print(df2['WORDtag'].value_counts().head(10)) #number of tags per POS tag
print(df2['WORDtag'].value_counts().sum()) #total number of POS tags

top10POStags = ['NN','NNP','IN','DT','JJ','NNS',',','VBD','.','_SP']
for tag in top10POStags:
    print(tag, ': \n')
    print(df2[df2['WORDtag'] == tag]['WORDtoken'].count() / num_tokens) #relative frequency
    print(df2[df2['WORDtag'] == tag]['WORDtag'].value_counts()) #Finegrained POS tags
    print(df2[df2['WORDtag'] == tag]['WORDtoken'].value_counts().head(3)) #3 most frequent token with this tag
    print(df2[df2['WORDtag'] == tag]['WORDtoken'].value_counts().tail(1)) #least frequent token with this tag

# 3. N-grams -----------------------------------------------------------------------------------------------------------
def get_ngrams(doc, n, filter_punct=False, use_pos=False):
    counter = 1
    ngrams = []
    el = ''
    for token in doc:
        if token.is_punct and filter_punct == True:
            continue
        if counter % n != 0:
            counter = counter + 1
            if use_pos:
                el = el + token.tag_ + ' '
            else:
                el = el + token.text + ' '
        else:
            counter = 1
            if use_pos:
                el = el + token.tag_
            else:
                el = el + token.text
            ngrams.append(el)
            el = ''
    return ngrams

bigrams = list(get_ngrams(doc, 2, filter_punct=False))
trigrams = list(get_ngrams(doc, 3, filter_punct=False))
bigrams_pos = list(get_ngrams(doc, 2, use_pos=True))
trigrams_pos = list(get_ngrams(doc, 3, use_pos=True))

# n-gram occurences
print(Counter(bigrams))
print(Counter(trigrams))
# pos n-gram occurences
print(Counter(bigrams_pos))
print(Counter(trigrams_pos))

# 4. Lemmatization -----------------------------------------------------------------------------------------------------
LEMtoken = []
LEMpos = []
LEMlem = []
for token in doc:
    if not token.is_punct: #exclude punctuation from lemmatization
        LEMtoken.append(token.text)
        LEMpos.append(token.pos_)
        LEMlem.append(token.lemma_)
df4 = pd.DataFrame({"LEMtoken": LEMtoken,"LEMpos": LEMpos,"LEMlem": LEMlem}) #all results in df4

df4[df4['LEMpos'] == 'VERB']['LEMlem'].value_counts().head(5) #find 5 verbs with common lemma which occurs often
print(df4[df4['LEMpos'] == 'VERB']['LEMlem'].value_counts().head(5))
inflections = df4[df4['LEMlem'] == 'say']['LEMtoken'].unique() #unique inflections of lemma 'say' in text
print(inflections)
for inflection in inflections:
    for line in all_sentences:
        if inflection in line.split(" "):
            print(line) #print the first line with a unique inflection
            break

# 5. NER  --------------------------------------------------------------------------------------------------------------
NERtoken = []
NERent = []
for ent in doc.ents:
    NERtoken.append(ent.text)
    NERent.append(ent.label_)
df5 = pd.DataFrame({"NERtoken": NERtoken,"NERent": NERent}) #all results in df5

print(df5.count()) #number of NERs
print(len(df5['NERtoken'].unique())) #number of different, unique NERs

print(all_sentences[:5]) #print first 5 sentences
print(df5.head(10)) #compare to the first 10 NERs, which includes all NERs from the first 5 sentences

