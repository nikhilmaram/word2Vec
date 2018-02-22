#!/usr/bin/env python

# import the required packages here
import numpy as np
import math
from word2vec import *

def run(text_path = 'text8',vocab_path = 'vocab.txt',output_path = 'vectors.txt'):

    ##print("Started Reading from vocab path")
    vocab = Vocabulary(vocab_path)
    ## Create a corpous list
    vocab.corpusWordsFunc(text_path)
    ## Create a dictionary of vocab word counts in corpus
    vocab.freqCorpusWords()
    ## Build a negative Table
    ##print("Building Negative table")
    vocab.buildNegativeTable()
    ##print("Building Network")
    net = network(vocab, embedDimension, contextSize, learningRate, negativeSampleSize, epochs)
    net.buildTable()
    ##print("Saving to File")
    saveFile(vocab.vocabWords, net.layer1,output_path)

