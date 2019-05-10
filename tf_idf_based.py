# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:58:14 2019

@author: mikedelong
"""
# https://medium.com/@acrosson/summarize-documents-using-tf-idf-bdee8f60b71
# https://github.com/acrosson/nlp/tree/master/summarization

# import os
import re
from json import load as json_load
from os.path import exists
from time import time

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from tika import parser

stop = stopwords.words('english')

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


def rank_sentences(arg_doc, arg_matrix, arg_features, top_n=3):
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
    sents = nltk.sent_tokenize(arg_doc)
    sentences = [nltk.word_tokenize(sent) for sent in sents]
    sentences = [[w for w in sent if nltk.pos_tag([w])[0][1] in NOUNS]
                 for sent in sentences]
    tfidf_sent = [[arg_matrix[arg_features.index(w.lower())]
                   for w in sent if w.lower() in arg_features]
                  for sent in sentences]

    # Calculate Sentence Values
    doc_val = sum(arg_matrix)
    sent_values = [sum(sent) / doc_val for sent in tfidf_sent]

    # Apply Similariy Score Weightings
    #    similarity_scores = [similarity_score(title, sent) for sent in sents]
    #    scored_sents = np.array(sent_values) + np.array(similarity_scores)

    # Apply Position Weights
    # ranked_sents = [sent * (i / len(sent_values)) for i, sent in enumerate(sent_values)]

    ranked_sents = [item for item in zip(range(len(sent_values)), sent_values)]
    ranked_sents = sorted(ranked_sents, key=lambda x: x[1] * -1)

    return ranked_sents[:top_n]


gutenberg_file_ids = ['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt',
                      'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt',
                      'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt',
                      'chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt',
                      'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt',
                      'shakespeare-macbeth.txt', 'whitman-leaves.txt']

if __name__ == '__main__':
    t0 = time()
    settings_file = 'tf_idf_based.json'
    with open(settings_file, 'r') as settings_fp:
        settings = json_load(settings_fp)

    data = list()
    nltk_corpus = settings['nltk_corpus'] if 'nltk_corpus' in settings.keys() else None
    if nltk_corpus is not None:
        if nltk_corpus == 'gutenberg':
            gutenberg_file_id = settings['gutenberg_file_id'] if 'gutenberg_file_id' in settings.keys() else None
            if gutenberg_file_id is not None and gutenberg_file_id in gutenberg_file_ids:
                data = list(nltk.corpus.gutenberg.words(gutenberg_file_id))
            else:
                print('nltk/gutenberg corpus file_id is missing from settings. Quitting.')
                quit(-1)
        else:
            print('setting {} must be {}. Quitting.'.format('nltk_corpus', 'gutenberg'))
            quit(-1)
    else:
        print('nltk_corpus is missing from settings. Quitting.')
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
    full_input_file = None
    if input_file is not None:
        full_input_file = input_folder + input_file if input_folder.endswith('/') else input_folder + '/' + input_file
    else:
        print('input file name is missing from settings. Quitting.')
        quit(-1)

    if not exists(full_input_file):
        print('input file {} is missing. Quitting.'.format(full_input_file))
        quit(-1)

    summary_size = settings['summary_size'] if 'summary_size' in settings.keys() else None
    if summary_size is not None:
        summary_size = int(summary_size)
    else:
        print('sentences to report missing from settings. Will report all sentences.')

    t1 = time()
    print('settings parse took {:5.2f}s'.format(t1 - t0))
    # Load the document you wish to summarize
    parsed = parser.from_file(full_input_file)
    document = parsed['content']
    t2 = time()
    print('target document read and parse took {:5.2f}s'.format(t2 - t1))
    cleaned_document = clean_document(document)
    t3 = time()
    print('document cleaning took {:5.2f}s'.format(t3 - t2))
    doc = remove_stop_words(cleaned_document)
    t4 = time()
    print('removing stopwords took {:5.2f}s'.format(t4 - t3))

    # Merge corpus data and new document data
    data = ' '.join(data)
    train_data = set(data.split() + doc.split())
    t5 = time()
    print('creating training data took {:5.2f}s'.format(t5 - t4))

    # Fit and Transform the term frequencies into a vector
    count_vect = CountVectorizer()
    count_vect = count_vect.fit(train_data)
    freq_term_matrix = count_vect.transform(train_data)
    feature_names = count_vect.get_feature_names()
    t6 = time()
    print('count vectorizer fitting and feature names took {:5.2f}s'.format(t6 - t5))

    # Fit and Transform the TfidfTransformer
    model = TfidfTransformer(norm='l2')
    model.fit(freq_term_matrix)
    t7 = time()
    print('fitting the TFIDF model took {:5.2f}s'.format(t7 - t6))

    # Get the dense tf-idf matrix for the document
    story_freq_term_matrix = count_vect.transform([doc])
    story_tfidf_matrix = model.transform(story_freq_term_matrix)
    story_dense = story_tfidf_matrix.todense()
    doc_matrix = story_dense.tolist()[0]
    t8 = time()
    print('making the document matrix took {:5.2f}s'.format(t8 - t7))

    # Get Top Ranking Sentences and join them as a summary
    top_sentences = rank_sentences(doc, doc_matrix, feature_names, top_n=summary_size)
    t9 = time()
    print('ranking sentences took {:5.2f}s'.format(t9 - t8))
    for pair in top_sentences:
        print('{} {}'.format(pair[1], '.\n'.join([cleaned_document.split('.')[pair[0]]])))
    tx = time()
    print('total run time {:5.2f}s'.format(tx - t0))

    # todo graph the whole tdidf curve
