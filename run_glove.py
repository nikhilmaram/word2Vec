#!/usr/bin/env python

# import the required packages here
import numpy as np
import math
from glove import *
## Define the parameters
learningRate = 0.01
contextSize = 5
embedDimension = 300
minCount = 5
numIterations = 1
maxCoOccur = 100
def run(text_path = 'text8',vocab_path = 'vocab.txt',output_path = 'vectors.txt'):

    ##print("Started Reading from vocab path")

    vocab = Vocabulary(vocab_path)
    vocabWords, vocabIndex = vocab.vocabWordsFunc()

    ## Create a corpous list
    vocab.corpusWordsFunc(text_path)
    ## Create a dictionary of vocab word counts in corpus
    corpusWords = vocab.freqCorpusWords()
    vocabDimension = len(vocabWords)

    coOccurenceMatrix = buildCoOcuuerenceMatrix(vocabWords, vocabIndex, corpusWords,contextSize=contextSize)

    ## Weights are precomputed from cooccurence matrix
    weightMatrix = preComputeWeights(coOccurenceMatrix, vocabDimension,maxCoOccur = maxCoOccur)

    expectedMatrix = expectedOutput(coOccurenceMatrix, vocabDimension,minCount=minCount)

    embedLayer, contextLayer = glove(vocabDimension=vocabDimension, coOccurenceMatrix=coOccurenceMatrix,
                                     weightMatrix=weightMatrix, expectedMatrix=expectedMatrix,numIter = numIterations,
                                     embedDimension=embedDimension,minCount=minCount,learningRate = learningRate)

    saveFile(vocabWords, embedLayer, output_path)

    ##print("saved")

if __name__ == '__main__':
    run()



