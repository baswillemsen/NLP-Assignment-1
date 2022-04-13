# Implement linguistic analyses using spacy
# Run them on data/preprocessed/train/sentences.txt
import spacy
from collections import Counter

nlp = spacy.load('en_core_web_sm')
input = open("sentences.txt", encoding="utf8")
lines = input.read()
input.close()
doc = nlp(lines)

## Tokanization
# Tokanization
word_frequencies = Counter()
num_sentences = 0
words_length = 0
for sentence in doc.sents:
    words = []
    num_sentences += 1
    for token in sentence:
        # Let's filter out punctuation
        if not token.is_punct:
            words.append(token.text)
            words_length = words_length + len(token)
    word_frequencies.update(words)

num_tokens = len(doc)
num_types = len(word_frequencies.keys())
num_words = sum(word_frequencies.values())
avg_words_per_sentence = sum(word_frequencies.values()) / num_sentences
avg_words_length = words_length / num_words

print(num_tokens, num_words, num_types, avg_words_per_sentence, avg_words_length)

