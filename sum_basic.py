# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:03:32 2019

@author: michael.delong
"""

# http://basicodingfordummies.blogspot.com/2015/12/sumbasic-algorithm-for-multi-document.html

from nltk.data import load
from nltk.tokenize import word_tokenize
from string import punctuation
from tika import parser
from time import time

'''
100-word summaries using SumBasic
'''

#Find word with highest probability in text
def highest_probability_word(wordprob_list):
    return wordprob_list[0][0]

#Find sentences containing highest probability word
def sentences_with_highest_pword(wordprob_list, sentences):
    subset_sent = []
    highest_pword = highest_probability_word(wordprob_list)
    i = 0
    for s in sentences:
        if (highest_pword in s):
            #Add weight of s to subset_sent
            tmp = weighted[i]
            subset_sent.append(tmp)
        else:
            subset_sent.append(0)
        i+=1
    return subset_sent

#Find highest scoring sentence from a subset
def highest_scoring_sentence(subset_sent):
    max_val, j, max_pos = 0, 0, 0
    for i in subset_sent:
        if (i > max_val):
            max_val = i
            max_pos = j
        j+=1
    return (max_val, max_pos)

#Calculate weight of each sentence
#weight = average probability of words in sentence
def cal_weight(sentences, wordprob, verbose=0):
    weight_sentences = {}
    i = 0
    for s in sentences:
        num_words = len(s.split())
        sum_up = 0
        for w in s.split():
            if w in wordprob.keys():
                sum_up += wordprob[w]
            else:
                if verbose > 0:
                    print('warning: we have no word probability for word {}'.format(w))
        sum_up = sum_up/num_words
        weight_sentences[i] = sum_up
        i+=1
    return weight_sentences

#Recalculate weight of each word in chosen sentence
#pnew = pold*pold
def recal_weight(chosen_sentence, wordprob):
    
    wordprob_new = wordprob
    for w in chosen_sentence.split():
        wordprob_new[w] = wordprob[w]*wordprob[w]
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
    input_file = './data/911-commission-full-report.pdf'
    parsed = parser.from_file(input_file)
    
    print('our content has {} characters'.format(len(parsed['content'])))
    contents = parsed['content']
    t1 = time()

    split_contents = contents.split()
    #Total number of words in the document
    N = len(split_contents)
    N1 = len(remove_punctuations(contents))
    print('our content has {} tokens and {} tokens without punctuation'.format(N, N1))

    #Storing probability of each word
    wordprob={}
    for word in split_contents:
        if word not in wordprob:
            wordprob[word] = 1 / float(N)
        else:
            wordprob[word] += 1 / float(N)
    
    #Sorting by maximum probability of words
    wordprob_list = sorted(wordprob.items(), key=lambda x: x[1], reverse=True) #sort by 2nd value, v
    
    #STEP 2
    #Dividing text into sentences
    summary = ''
    tokenizer = load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(contents)
    t2 = time()
    
    #Loop till number of words in the summary is less than 100
    while len(remove_punctuations(summary)) <= 200: 
        
        weighted = cal_weight(sentences, wordprob)    
        #STEP 3: Pick the best scoring sentence {that contains the highest probability word}
        #3.1: Pick sentences that contain the highest probability word
        subset_sent = sentences_with_highest_pword(wordprob_list, sentences)
        #3.2: Pick the highest scoring sentence from 3.1
        max_val, max_pos = highest_scoring_sentence(subset_sent)

        #STEP 4: Updating the probability of each word in the chosen sentence
        chosen_sentence = sentences[max_pos]
        summary = summary + chosen_sentence + ' '
        sentences.remove(chosen_sentence)
        if max_pos in weighted:        
            del weighted[max_pos]
        wordprob_list = wordprob.items()
        #Resorting the list of word probabilities
        wordprob_list = sorted(wordprob_list, key=lambda x: x[1], reverse=True) 
        
    t3 = time()
    print ('Here is our summary:\n{}'.format(summary.strip()))
    print ('Our summary is {} characters'.format(len(summary)))
    print ('Our summary is {} tokens'.format(len(remove_punctuations(summary))))
    print('raw data load time is {:5.2f}s'.format(t1 - t0))
    print('tokenize time is {:5.2f}s'.format(t2 - t1))
    print('summary time is {:5.2f}s'.format(t3 - t2))
    print('total time is {:5.2f}s'.format(t3 - t0))
