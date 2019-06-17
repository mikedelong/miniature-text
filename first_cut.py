# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:13:06 2019

@author: mikedelong
"""

from json import load
from os.path import exists
from time import time

import matplotlib.pyplot as plt
from nltk import FreqDist
from nltk import word_tokenize
from nltk.tokenize import casual_tokenize
from nltk.tokenize import sent_tokenize
from tika import parser


def fix(arg):
    result = list()
    skip = False
    for item_index, item in enumerate(arg):
        if item.endswith('-'):
            out_item = item[:-1]
            result.append(out_item + arg[item_index + 1])
            skip = True
        else:
            if skip:
                skip = False
            else:
                result.append(item)
    return result


settings_file = 'first_cut.json'
if __name__ == '__main__':
    time_start = time()
    with open(settings_file, 'r') as settings_fp:
        settings = load(settings_fp)

    input_folder = settings['input_folder'] if \
        'input_folder' in settings.keys() else None
    if input_folder is not None:
        if not exists(input_folder):
            print('input data folder does not exist. Quitting.')
            quit(-1)
    else:
        print('input folder name is missing from settings. Quitting.')
        quit(-1)

    input_file = settings['input_file'] if 'input_file' in settings.keys() else None
    full_input_file = None
    if input_file is not None:
        full_input_file = input_folder + input_file if input_folder.endswith('/') else input_folder + '/' + input_file
    else:
        print('input file name is missing from settings. Quitting.')
        quit(-1)

    if not exists(full_input_file):
        print('input file {} is missing. Quitting.'.format(full_input_file))
        quit(-1)

    parsed = parser.from_file(full_input_file)

    print('our content is a string of length {}'.format(len(parsed['content'])))
    content = parsed['content']

    # todo fix quotes
    # todo put summary sentences back in order
    sentences = sent_tokenize(content, language='english')

    print('before we remove very short sentences we have {} sentences'.format(len(sentences)))
    sentences = [item for item in sentences if len(item) > 1]
    print('after we remove very short sentences we have {} sentences'.format(len(sentences)))

    running_count = 0
    for index_sentence, sentence in enumerate(sentences):
        words = casual_tokenize(sentence)
        for index, word in enumerate(words):
            if word.endswith('-'):
                running_count += 1
                print('{}: {} : {}'.format(index_sentence, running_count, words[index - 1] + word + words[index + 1]))

    modified = list()
    for index_sentence, sentence in enumerate(sentences):
        words = word_tokenize(sentence)
        fixed_words = fix(words)
        modified.append(' '.join(fixed_words))
    sentences = modified

    # now we need to count words and do a frequency score for each word
    freq_dist = FreqDist()
    for sentence in sentences:
        for word in word_tokenize(sentence):
            freq_dist[word.lower()] += 1

    word_count = freq_dist.N()
    print('our frequency distribution has {} items'.format(len(freq_dist)))
    print('our word count is {}'.format(word_count))

    # now let's traverse our frequency distribution and build our token-probability map
    probabilities = {word: float(freq_dist[word]) / float(word_count) for word in freq_dist}

    # now we are ready to calculate scores for each sentence
    sentence_scores = {
        sentence: (sum([probabilities[word.lower()] for word in word_tokenize(sentence)]), index)
        for index, sentence in enumerate(sentences)}

    candidates = {key: value for key, value in sentence_scores.items() if value[0] > 3.0}

    for key, value in candidates.items():
        print('{:5.2f} {} {}'.format(value[0], value[1], key.replace('\n', ' ')))

    values = [value for value in candidates.values()]
    plt.plot(sorted(values), 'o', mfc='none')
    output_file = './output/first_cut_plot.png'
    print('writing score plot to {}'.format(output_file))
    plt.savefig(output_file)
    print('total elapsed time: {:5.2f}s'.format(time() - time_start))
