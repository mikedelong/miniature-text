# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:03:32 2019

@author: mikedelong
"""

# http://basicodingfordummies.blogspot.com/2015/12/sumbasic-algorithm-for-multi-document.html

from string import punctuation
from time import time

from nltk.data import load
from nltk.tokenize import word_tokenize
from tika import parser

'''
100-word summaries using SumBasic
'''


# Find word with highest probability in text
def highest_probability_word(arg):
    return arg[0][0]


# Find sentences containing highest probability word
def sentences_with_highest_pword(arg_word_probability_list, arg_sentences):
    highest = highest_probability_word(arg_word_probability_list)
    i = 0
    # todo use a comprehension here?
    result = []
    for sentence in arg_sentences:
        if highest in sentence:
            # Add weight of s to subset_sent
            tmp = weighted[i]
            result.append(tmp)
        else:
            result.append(0)
        i += 1
    return result


# Find highest scoring sentence from a subset
def highest_scoring_sentence(arg_subset):
    result_value, j, result_position = 0, 0, 0
    for i in arg_subset:
        if i > result_value:
            result_value = i
            result_position = j
        j += 1
    return result_value, result_position


# Calculate weight of each sentence
# weight = average probability of words in sentence
def cal_weight(arg_sentences, arg_word_probability, verbose=0):
    result = {}
    # todo use enumerate here
    i = 0
    for sentence in arg_sentences:
        word_count: int = len(sentence.split())
        sum_up = 0
        for local_word in sentence.split():
            if local_word in arg_word_probability.keys():
                sum_up += arg_word_probability[local_word]
            else:
                if verbose > 0:
                    print('warning: we have no word probability for word {}'.format(local_word))
        sum_up = float(sum_up) / float(word_count)
        result[i] = sum_up
        i += 1
    return result


# Recalculate weight of each word in chosen sentence
# pnew = pold*pold
def recalculate_weight(arg_sentence, arg_word_probability):
    wordprob_new = arg_word_probability
    for local_word in arg_sentence.split():
        wordprob_new[local_word] = arg_word_probability[local_word] * arg_word_probability[local_word]
    return wordprob_new


our_punctuation = list(punctuation) + ['\n', '\r', '\r\n']

'''Removes all punctuations from a given string and returns a list 
   containing the words in the given string '''


def remove_punctuations(summary):
    tokenized_summary = word_tokenize(summary)
    result = [word for word in tokenized_summary if word not in our_punctuation]
    return result


if __name__ == '__main__':
    # todo think about moving input identification to a config file
    #    num_args = len(sys.argv)
    #    file_name = sys.argv[1]
    #    fp = open(file_name)
    #    contents = fp.read()

    t0 = time()
    input_file = './data/911Report.pdf'
    parsed = parser.from_file(input_file)

    print('our content has {} characters'.format(len(parsed['content'])))
    contents = parsed['content']
    t1 = time()

    split_contents = contents.split()
    # Total number of words in the document
    N = len(split_contents)
    N1 = len(remove_punctuations(contents))
    print('our content has {} tokens and {} tokens without punctuation'.format(N, N1))

    # Storing probability of each word
    wordprob = {}
    for word in split_contents:
        if word not in wordprob:
            wordprob[word] = 1 / float(N)
        else:
            wordprob[word] += 1 / float(N)

    # Sorting by maximum probability of words
    wordprob_list = sorted(wordprob.items(), key=lambda x: x[1], reverse=True)  # sort by 2nd value, v

    # STEP 2
    # Dividing text into sentences
    summary = ''
    tokenizer = load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(contents)
    t2 = time()

    # Loop till number of words in the summary is less than 100
    while len(remove_punctuations(summary)) <= 200:

        weighted = cal_weight(sentences, wordprob)
        # STEP 3: Pick the best scoring sentence {that contains the highest probability word}
        # 3.1: Pick sentences that contain the highest probability word
        subset_sent = sentences_with_highest_pword(wordprob_list, sentences)
        # 3.2: Pick the highest scoring sentence from 3.1
        max_val, max_pos = highest_scoring_sentence(subset_sent)

        # STEP 4: Updating the probability of each word in the chosen sentence
        chosen_sentence = sentences[max_pos]
        summary = summary + chosen_sentence + ' '
        sentences.remove(chosen_sentence)
        if max_pos in weighted:
            del weighted[max_pos]
        wordprob_list = wordprob.items()
        # Resorting the list of word probabilities
        wordprob_list = sorted(wordprob_list, key=lambda x: x[1], reverse=True)

    t3 = time()
    print('Here is our summary:\n{}'.format(summary.strip()))
    print('Our summary is {} characters'.format(len(summary)))
    print('Our summary is {} tokens'.format(len(remove_punctuations(summary))))
    print('raw data load time is {:5.2f}s'.format(t1 - t0))
    print('tokenize time is {:5.2f}s'.format(t2 - t1))
    print('summary time is {:5.2f}s'.format(t3 - t2))
    print('total time is {:5.2f}s'.format(t3 - t0))
