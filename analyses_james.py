# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt

import spacy
import pandas as pd
import numpy as np
import nltk
from collections import Counter
from nltk import word_tokenize
from nltk.util import ngrams

nlp = spacy.load("en_core_web_sm")

# get sentences dataset
file_path = 'data/preprocessed/train/sentences.txt'
sentences = open(file_path, encoding='utf-8').read()

# get nlp pipeline (tokenizer, tagger, parser, named-entity recog, etc) by creating nlp object
doc = nlp(sentences)


# 1. Tokanization ------------------------------------------------------------------------------------------------------
# Extract tokens for the given doc and add in list
tokens = [token for token in doc]

# Extract token text
tokens_text = [token.text for token in doc]

# number of tokens
NO_OF_TOKENS = len(tokens_text)
print(NO_OF_TOKENS)

# number of types (unique tokens)
print(len(set(tokens_text)))

# number of words
word_tokens = [token for token in doc if not token.is_punct and token.text != '\n']
print(len(word_tokens))

# avg number of words per sentence
count = 0
for sentence in doc.sents:
    count = count + 1
    #print(sentence)

avg_per_sentence = len(word_tokens)/count
print(avg_per_sentence)

# avg word length
word_length = [len(word.text) for word in word_tokens]
print(np.mean(word_length))


# 2. Word Classes ------------------------------------------------------------------------------------------------------
# Fine-grained POS tags
pos_tags = [token.tag_ for token in tokens]
#print(pos_tags)
top_ten = Counter(pos_tags)
print(top_ten)

# Get universal POS tags
def get_universal_pos(tokens, fine):
    uni_tags = [token.pos_ for token in tokens if token.tag_ == fine]
    return uni_tags

# search by fine-grained (for multiple universal tags, just do counter on specified fine-grained list and got top occurence)
uni = get_universal_pos(tokens, 'VBN')
# print(uni)
print(Counter(uni))

# relative tag frequency
def rel_tag_freq(fine_tag_freq, no_of_tokens):
    return fine_tag_freq/no_of_tokens

# NN relative tag frequency example
print(rel_tag_freq(2066, NO_OF_TOKENS))

# Get most frequent words by fine and uni tags
def most_frequent_words(tokens, fine, uni):
    words = [token.text for token in tokens if token.tag_ == fine and token.pos_ == uni]
    return Counter(words)

# most freq words examples
#print(most_frequent_words(tokens, 'DT', 'DET'))
#print(most_frequent_words(tokens, 'JJ', 'ADJ'))
print(most_frequent_words(tokens, 'NNS', 'NOUN'))
#print(most_frequent_words(tokens, ',', 'PUNCT'))
#print(most_frequent_words(tokens, '.', 'PUNCT'))
#print(most_frequent_words(tokens, '_SP', 'SPACE'))
#print(most_frequent_words(tokens, 'VBN', 'VERB'))


# 3. N-grams -----------------------------------------------------------------------------------------------------------
# n-grams using textacy did not work well
#bigrams = list(te.extract.basics.ngrams(doc, 2, filter_punct=True))
#trigrams = list(te.extract.basics.ngrams(doc, 3, filter_punct=True))

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

# n-gram occurences
print(Counter(bigrams))
print(Counter(trigrams))

bigrams_pos = list(get_ngrams(doc, 2, use_pos=True))
trigrams_pos = list(get_ngrams(doc, 3, use_pos=True))

# pos n-gram occurences
print(Counter(bigrams_pos))
print(Counter(trigrams_pos))


# 5. NERs --------------------------------------------------------------------------------------------------------------
# number of NERs
ents = [ent.text for ent in doc.ents]
print(len(ents))

# number of unique NERs
ent_labels = [ent.label_ for ent in doc.ents]
print(len(set(ents)))

# more NER info
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)