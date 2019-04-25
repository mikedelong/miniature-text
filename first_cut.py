# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:13:06 2019

@author: mikedelong
"""

import json
from os import makedirs
from os.path import exists

import matplotlib.pyplot as plt
import nltk.data
from nltk import FreqDist
from nltk import word_tokenize
from tika import parser

settings_file = 'first_cut.json'
with open(settings_file, 'r') as settings_fp:
    settings = json.load(settings_fp)

tokenizer_pickle = settings['tokenizer_pickle'] if 'tokenizer_pickle' in settings.keys() else None
if tokenizer_pickle is not None:
    tokenizer = nltk.data.load(tokenizer_pickle)
else:
    print('tokenizer pickle is not defined in settings file {}. Quitting.'.format(settings_file))
    quit(-1)

input_folder = settings['input_folder'] if 'input_folder' in settings.keys() else None
if input_folder is not None:
    if not exists(input_folder):
        print('input data folder does not exist. Quitting.')
        quit(-1)
else:
    print('input folder name is missing from settings. Quitting.')
    quit(-1)

input_file = settings['input_file'] if 'input_file' in settings.keys() else None
if input_file is not None:
    full_input_file = input_folder + input_file  # what about /?
else:
    print('input file name is missing from settings. Quitting.')
    quit(-1)

if not exists(full_input_file):
    print('input file {} is missing. Quitting.'.format(full_input_file))
    quit(-1)

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
