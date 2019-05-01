# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:58:14 2019

@author: mikedelong
"""
# https://medium.com/@acrosson/summarize-documents-using-tf-idf-bdee8f60b71
# https://github.com/acrosson/nlp/tree/master/summarization

# import os
import re

import nltk
# import numpy as np
# import datetime
# import xml.etree.ElementTree as ET
# from bs4 import BeautifulSoup
from nltk.corpus import stopwords

stop = stopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from json import load as json_load
from os.path import exists
from time import time

from tika import parser

# Noun Part of Speech Tags used by NLTK
# More can be found here
# http://www.winwaed.com/blog/2011/11/08/part-of-speech-tags/
NOUNS = ['NN', 'NNS', 'NNP', 'NNPS']


def clean_document(arg):
    """Cleans document by removing unnecessary punctuation. It also removes
    any extra periods and merges acronyms to prevent the tokenizer from
    splitting a false sentence

    """
    # Remove all characters outside of Alpha Numeric
    # and some punctuation
    arg = re.sub('[^A-Za-z .-]+', ' ', arg)
    arg = arg.replace('-', '')
    arg = arg.replace('...', '')
    arg = arg.replace('Mr.', 'Mr').replace('Mrs.', 'Mrs')

    # Remove acronyms M.I.T. -> MIT
    # to help with sentence tokenizing
    arg = merge_acronyms(arg)

    # Remove extra whitespace
    arg = ' '.join(arg.split())
    return arg


def remove_stop_words(arg):
    """Returns document without stop words"""
    arg = ' '.join([i for i in arg.split() if i not in stop])
    return arg


def similarity_score(t, s):
    """Returns a similarity score for a given sentence.

    similarity score = the total number of tokens in a sentence that exits
                        within the title / total words in title

    """
    t = remove_stop_words(t.lower())
    s = remove_stop_words(s.lower())
    t_tokens, s_tokens = t.split(), s.split()
    similar = [w for w in s_tokens if w in t_tokens]
    score = (len(similar) * 0.1) / len(t_tokens)
    return score


def merge_acronyms(s):
    """Merges all acronyms in a given sentence. For example M.I.T -> MIT"""
    r = re.compile(r'(?:(?<=\.|\s)[A-Z]\.)+')
    acronyms = r.findall(s)
    for a in acronyms:
        s = s.replace(a, a.replace('.', ''))
    return s


def rank_sentences(doc, doc_matrix, feature_names, top_n=3):
    """Returns top_n sentences. Theses sentences are then used as summary
    of document.

    input
    ------------
    doc : a document as type str
    doc_matrix : a dense tf-idf matrix calculated with Scikits TfidfTransformer
    feature_names : a list of all features, the index is used to look up
                    tf-idf scores in the doc_matrix
    top_n : number of sentences to return

    """
    sents = nltk.sent_tokenize(doc)
    sentences = [nltk.word_tokenize(sent) for sent in sents]
    sentences = [[w for w in sent if nltk.pos_tag([w])[0][1] in NOUNS]
                 for sent in sentences]
    tfidf_sent = [[doc_matrix[feature_names.index(w.lower())]
                   for w in sent if w.lower() in feature_names]
                  for sent in sentences]

    # Calculate Sentence Values
    doc_val = sum(doc_matrix)
    sent_values = [sum(sent) / doc_val for sent in tfidf_sent]

    # Apply Similariy Score Weightings
    #    similarity_scores = [similarity_score(title, sent) for sent in sents]
    #    scored_sents = np.array(sent_values) + np.array(similarity_scores)

    # Apply Position Weights
    # ranked_sents = [sent * (i / len(sent_values)) for i, sent in enumerate(sent_values)]

    ranked_sents = [pair for pair in zip(range(len(sent_values)), sent_values)]
    ranked_sents = sorted(ranked_sents, key=lambda x: x[1] * -1)

    return ranked_sents[:top_n]


if __name__ == '__main__':
    # todo: add data pickle
    # todo: move this data file name to a settings file
    # Load corpus data used to train the TF-IDF Transformer
    # data = pickle.load(open('./data/data.pkl', 'rb'))
    # data = list(nltk.corpus.gutenberg.words('austen-emma.txt'))
    # todo use a real corpus here
    data = ['this and that', 'the other thing']

    # Load the document you wish to summarize
    title = ''
    document = ''

    settings_file = 'tf_idf_based.json'
    with open(settings_file, 'r') as settings_fp:
        settings = json_load(settings_fp)

    input_folder = settings['input_folder'] if 'input_folder' in settings.keys() else None
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

    # todo time the major sections of the code and report incremental timing results
    t0 = time()
    parsed = parser.from_file(full_input_file)

    document = parsed['content']

    cleaned_document = clean_document(document)
    doc = remove_stop_words(cleaned_document)

    # Merge corpus data and new document data
    # data = [' '.join(document) for document in data]
    data = ' '.join(data)

    train_data = set(data.split() + doc.split())

    # Fit and Transform the term frequencies into a vector
    count_vect = CountVectorizer()
    count_vect = count_vect.fit(train_data)
    freq_term_matrix = count_vect.transform(train_data)
    feature_names = count_vect.get_feature_names()

    # Fit and Transform the TfidfTransformer
    tfidf = TfidfTransformer(norm="l2")
    tfidf.fit(freq_term_matrix)

    # Get the dense tf-idf matrix for the document
    story_freq_term_matrix = count_vect.transform([doc])
    story_tfidf_matrix = tfidf.transform(story_freq_term_matrix)
    story_dense = story_tfidf_matrix.todense()
    doc_matrix = story_dense.tolist()[0]

    # Get Top Ranking Sentences and join them as a summary
    top_sents = rank_sentences(doc, doc_matrix, feature_names)
    # todo report the summary sentences on separate lines
    # todo report scores(?) for the summary sentences
    summary = '.'.join([cleaned_document.split('.')[i]
                        for i in [pair[0] for pair in top_sents]])
    summary = ' '.join(summary.split())
    print(summary)
    # todo report total time

