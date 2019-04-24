# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:13:06 2019

@author: mikedelong
"""

import matplotlib.pyplot as plt
import nltk.data
from nltk import FreqDist
from nltk import word_tokenize
from tika import parser

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

input_file = './data/911Report.pdf'
parsed = parser.from_file(input_file)

print('our content is a string of length {}'.format(len(parsed['content'])))
content = parsed['content']

sentences = tokenizer.tokenize(content)
print('before we remove very short sentences we have {} sentences'.format(len(sentences)))
sentences = [item for item in sentences if len(item) > 1]
print('after we remove very short sentences we have {} sentences'.format(len(sentences)))

# now we need to count words and do a frequency score for each word

freq_dist = FreqDist()
for sentence in sentences:
    for word in word_tokenize(sentence):
        freq_dist[word.lower()] += 1

word_count = freq_dist.N()
print('our frequency distribution has {} items'.format(len(freq_dist)))
print('our word count is {}'.format(freq_dist.N()))

# now let's traverse our frequency distribution and build our token-probability map
probabilities = {
    word: float(freq_dist[word]) / float(word_count) for word in freq_dist}

# now we are ready to calculate scores for each sentence
sentence_scores = {
    sentence: sum([probabilities[word.lower()] for word in word_tokenize(sentence)])
    for sentence in sentences}

candidates = {key: value for key, value in sentence_scores.items() if value > 3.0}

for key, value in candidates.items():
    print('{:5.2f} {}'.format(value, key[:100]))

values = [value for value in candidates.values()]
plt.plot(sorted(values))
plt.show()
