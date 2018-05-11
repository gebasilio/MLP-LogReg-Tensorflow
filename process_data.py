from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import random
import os
import sys
sys.path.append('..')
import time
from pymongo import MongoClient
from datetime import datetime
import nltk
nltk.download('perluniprops')
nltk.download('nonbreaking_prefixes')
nltk.download('stopwords')
from nltk.tokenize.moses import MosesTokenizer
import numpy as np
from six.moves import urllib
from nltk.corpus import stopwords

import utils

# Parameters for downloading data
DOWNLOAD_URL = 'http://mattmahoney.net/dc/'
EXPECTED_BYTES = 31344016
DATA_FOLDER = 'data/'
FILE_NAME = 'text8.zip'

def download(file_name, expected_bytes):
    """ Download the dataset text8 if it's not already downloaded """
    file_path = DATA_FOLDER + file_name
    if os.path.exists(file_path):
        print("Dataset ready")
        return file_path
    file_name, _ = urllib.request.urlretrieve(DOWNLOAD_URL + file_name, file_path)
    file_stat = os.stat(file_path)
    if file_stat.st_size == expected_bytes:
        print('Successfully downloaded the file', file_name)
    else:
        raise Exception('File ' + file_name +
                        ' might be corrupted. You should try downloading it with a browser.')
    return file_path


def build_vocab(words, vocab_size):
    """ Build vocabulary of VOCAB_SIZE most frequent words """
    dictionary = dict()
    count = [('UNK', -1)]
    count.extend(Counter(words).most_common(vocab_size - 1))
    index = 0
    utils.make_dir('processed')
    with open('processed/vocab_1000.tsv', "w") as f:
        for word, _ in count:
            dictionary[word] = index
            if index < 1000:
                f.write(word + "\n")
            index += 1
    index_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, index_dictionary

def convert_words_to_index(words, dictionary):
    """ Replace each word in the dataset with its index in the dictionary """
    return [[dictionary[word] if word in dictionary else 0 for word in article] for article in words]

def generate_sample(index_words, context_window_size):
    while True:
        for article in index_words:
            """ Form training pairs according to the skip-gram model. """
            for index, center in enumerate(article):
                context = random.randint(1, context_window_size)
                # get a random target before the center word
                for target in article[max(0, index - context): index]:
                    yield center, target
                # get a random target after the center wrod
                for target in article[index + 1: index + context + 1]:
                    yield center, target

def get_batch(iterator, batch_size):
    """ Group a numerical stream into batches and yield them as Numpy arrays. """
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(iterator)
        yield center_batch, target_batch

def extract_data_from_db(start_year, stop_year, strategy="article"):
    print("hola")
    stopWords = set(stopwords.words('english'))
    # Posibles estrategias: article, sentence
    client = MongoClient()
    db = client.nyt
    collection=db["caratulas"]
    start_date = datetime(start_year, 1, 1, 0, 0, 0)
    end_date = datetime(stop_year, 12, 31, 23, 59, 59)
    cursor=collection.find( {"$and":[{ "lead_paragraph": { "$exists": True, "$nin": [None]}} , 
                                 {"pub_date":{"$exists":True, "$lt":end_date,"$gte":start_date}}]})
    articles=[x["lead_paragraph"].lower() for x in cursor]
    if(strategy=="article"):
        tokenizer=MosesTokenizer()
        articles_tok=[[w for w in tokenizer.tokenize(x) if w not in stopWords and w.isalpha()] for x in articles]
    elif(strategy=="sentence"):
        tokenizer=MosesTokenizer()
        articles_tok=[[w for w in tokenizer.tokenize(y) if w.isalpha()] for x in articles for y in x.split(". ") ]
        
    return articles_tok
        
def process_data(vocab_size, batch_size, skip_window):
    client = MongoClient()
    db = client.nyt
    collection=db["caratulas"]
    start_date = datetime(2016, 1, 1, 0, 0, 0)
    end_date = datetime(2017, 1, 1, 0, 0, 0)
    cursor=collection.find( {"$and":[{ "lead_paragraph": { "$exists": True, "$nin": [None]}} , 
                                 {"pub_date":{"$exists":True, "$lt":end_date,"$gte":start_date}}]})
    articles=[x["lead_paragraph"].lower() for x in cursor]
    tokenizer=MosesTokenizer()
    articles_tok=[tokenizer.tokenize(x) for x in articles]
    flat_art=[x for article in articles_tok for x in article]
    dictionary, _ = build_vocab(flat_art, vocab_size)
    index_words = convert_words_to_index(articles_tok, dictionary)
    del flat_art # to save memory
    del articles_tok
    single_gen = generate_sample(index_words, skip_window)
    return get_batch(single_gen, batch_size)

def get_index_vocab(vocab_size):
    file_path = download(FILE_NAME, EXPECTED_BYTES)
    words = read_data(file_path)
    return build_vocab(words, vocab_size)


def culo():
    print("hola")