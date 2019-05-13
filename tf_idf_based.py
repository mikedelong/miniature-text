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


# todo think about caching the t_tokens instead of regenerating them
def similarity_score(t, s):
    """Returns a similarity score for a given sentence.

    similarity score = the total number of tokens in a sentence that exist
                        within the title / total words in title

    """
    t_tokens = remove_stop_words(t.lower()).split()
    s_tokens = remove_stop_words(s.lower()).split()
    similar = [1 for word in s_tokens if word in t_tokens]
    result = float(sum(similar)) / (10.0 * float(len(t_tokens)))
    return result


acronym_regex = re.compile(r'(?:(?<=\.|\s)[A-Z]\.)+')


def merge_acronyms(arg):
    """Merges all acronyms in a given sentence. For example M.I.T -> MIT"""
    acronyms = acronym_regex.findall(arg)
    for acronyms in acronyms:
        arg = arg.replace(acronyms, acronyms.replace('.', ''))
    return arg


def get_sentences(arg_doc):
    local_sentences = nltk.sent_tokenize(arg_doc)
    result = [nltk.word_tokenize(sentence) for sentence in local_sentences]
    return result


def rank_sentences(arg_sentences, arg_matrix, arg_features, top_n=3):
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
    times = [time()]
    local_sentences = [[word for word in sentence if nltk.pos_tag([word])[0][1] in NOUNS] for sentence in arg_sentences]
    times.append(time())
    print('rank_sentences: marking nouns took {:5.2f}s'.format(times[-1] - times[-2]))
    # todo can we use something from sklearn here?
    tdidf_sentences = [[arg_matrix[arg_features.index(word.lower())] for word in sentence if word.lower()
                        in arg_features] for sentence in local_sentences]
    times.append(time())
    print('rank_sentences: manual TFIDF took {:5.2f}s'.format(times[-1] - times[-2]))

    # Calculate Sentence Values
    doc_val = sum(arg_matrix)
    times.append(time())
    print('rank_sentences: summing the arg_matrix took {:5.2f}s'.format(times[-1] - times[-2]))

    sent_values = [sum(sent) / doc_val for sent in tdidf_sentences]
    times.append(time())
    print('rank_sentences: calculating sentence values took {:5.2f}s'.format(times[-1] - times[-2]))

    # Apply Similarity Score Weightings
    #    similarity_scores = [similarity_score(title, sent) for sent in sents]
    #    scored_sents = np.array(sent_values) + np.array(similarity_scores)

    # Apply Position Weights
    # ranked_sents = [sent * (i / len(sent_values)) for i, sent in enumerate(sent_values)]

    ranked = [item for item in zip(range(len(sent_values)), sent_values)]
    times.append(time())
    print('rank_sentences: ranking sentences took {:5.2f}s'.format(times[-1] - times[-2]))
    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
    times.append(time())
    print('rank_sentences: sorting ranked sentences took {:5.2f}s'.format(times[-1] - times[-2]))

    return ranked[:top_n]


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

    sentences = get_sentences(doc)
    t9 = time()
    print('getting sentences took {:5.2f}s'.format(t9 - t8))

    # Get Top Ranking Sentences and join them as a summary
    top_sentences = rank_sentences(sentences, doc_matrix, feature_names, top_n=summary_size)
    tix = time()
    print('ranking sentences took {:5.2f}s'.format(tix - t9))
    for pair in top_sentences:
        print('{} {}'.format(pair[1], '.\n'.join([cleaned_document.split('.')[pair[0]]])))
    tx = time()
    print('total run time {:5.2f}s'.format(tx - t0))

    # todo graph the whole tdidf curve
