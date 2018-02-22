import numpy as np
import math

embedDimension = 300
contextSize = 6
learningRate = 0.01
negativeSampleSize = 6
epochs = 20

def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def neuralNets(vocabDimension, embedDimension):
    ## First layer is initailised such that variance of input is maintained.
    layer1 = np.random.uniform(low=-0.5 / embedDimension, high=0.5 / embedDimension,
                               size=(vocabDimension, embedDimension))
    ## Second layer is initialised with all zeroes
    ## layer 2 transpose is considered here because the training time decreases when accessed by row than column.
    layer2 = np.zeros(shape=(vocabDimension, embedDimension))
    return layer1, layer2

def saveFile(vocabWords,layer,fileName):
    f = open(fileName,'w')
    for i in range(len(vocabWords)):
        s = ""
        s = str(vocabWords[i])
        for j in range(300):
            s = s + " " + str(layer[i][j])
        s = s +"\n"
        f.write(s)

## Vocabulary class
class Vocabulary:
    def __init__(self, vocabFile):
        self.vocabFile = vocabFile
        self.vocabWords = []
        self.vocabToIndex = {}
        self.corpusWordCount = {}
        self.vocabWordsFunc()

    ## Creates a Vocabulary list from the given vocabulary File
    def vocabWordsFunc(self):
        f = open(self.vocabFile, 'r')
        i = 0
        for l in f:
            word = l.strip('\n')
            self.vocabToIndex[word] = len(self.vocabWords)
            self.vocabWords.append(word)
            self.corpusWordCount[word] = 0
        self.vocabWordsSet = set(x for x in self.vocabWords)

    ## Creates a corpus list from the given corpus File
    def corpusWordsFunc(self, corpusFile):
        f = open(corpusFile, 'r')
        self.corpusWords = []
        for l in f:
            self.corpusWords.extend(l.split())

    ## Creates a dictionary which maintains count of each vocab word in corpus
    def freqCorpusWords(self):
        for index in range(len(self.corpusWords)):
            word = self.corpusWords[index]
            if word in self.vocabWordsSet:
                ## Adds one to existing count
                self.corpusWordCount[word] = self.corpusWordCount[word] + 1
            else:
                self.corpusWords[index] = 'UNKNOWN'

    ## Creates a table for getting neagative samples
    def buildNegativeTable(self):
        ## According to word2Vec implementation, most frequent words are considered to be negative samples.
        ## An array of size 100M are considered and each word is filled with normalised probability that it
        ## occurs times the size of the table
        power = 0.75
        norm = np.sum(np.power(self.corpusWordCount[word], power) for word in self.corpusWordCount if word != 'UNKNOWN')
        ## Create a negTable
        self.negTableSize = np.power(10, 8)
        self.negTable = np.zeros(self.negTableSize, dtype=np.int32)
        i = 0
        prob = 0
        ## Fill the negative table with word Index
        for word, count in self.corpusWordCount.items():
            if (word != 'UNKNOWN'):
                prob = prob + (np.power(count, power) / norm)
                wordIndex = self.vocabToIndex[word]
                while i < self.negTableSize and float(i) / self.negTableSize < prob:
                    self.negTable[i] = wordIndex
                    i += 1

    def sampleNegative(self, count, seed):
        np.random.seed(seed)
        indices = np.random.randint(low=0, high=self.negTableSize, size=count)
        return [self.vocabWords[self.negTable[i]] for i in indices]




class network:
    def __init__(self, vocab, embedDimension, contextSize, learningRate, negativeSampleSize, epochs):

        self.embedDimension = embedDimension

        self.contextSize = contextSize
        self.negativeSampleSize = negativeSampleSize
        self.epochs = epochs

        self.learningRate = learningRate

        ## Create the vocabulary
        self.vocab = vocab
        self.vocabDimension = len(self.vocab.vocabWords)

        ## creating the two layers and randomly initializing them
        self.layer1, self.layer2 = neuralNets(self.vocabDimension, self.embedDimension)

    def buildTable(self):
        alpha = self.learningRate
        currNegSampleSize = self.contextSize * self.negativeSampleSize
        prevCost = 0
        currCost = 0
        for i in range(epochs):
            currCost = 0
            for index in range(self.contextSize, len(self.vocab.corpusWords) - self.contextSize):
                centerWord = self.vocab.corpusWords[index]
                if centerWord != 'UNKNOWN':
                    centerIndex = self.vocab.vocabToIndex[centerWord]

                    ##contextStart = max(0,index-self.contextSize)
                    ##contextEnd = min(index+self.contextSize+1,len(self.vocab.corpusWords))
                    contextStart = index - self.contextSize
                    contextEnd = index + self.contextSize + 1
                    context = self.vocab.corpusWords[contextStart:index] + self.vocab.corpusWords[
                                                                           contextStart + 1:contextEnd]

                    ## Get the negative samples
                    negSamples = self.vocab.sampleNegative(currNegSampleSize, index)
                    layer1S = self.layer1[centerIndex]
                    summation = np.zeros(embedDimension)
                    currError = 1
                    positiveClassifiers = [(contextWord, 1) for contextWord in context]
                    negativeClassifiers = [(negWord, 0) for negWord in negSamples]

                    ## Doing negative and positive separately because cost function can be calculated easily

                    for classifierWord, value in positiveClassifiers:
                        if classifierWord != 'UNKNOWN':
                            classifierIndex = self.vocab.vocabToIndex[classifierWord]
                            layer2C = self.layer2[classifierIndex]
                            z = np.dot(layer1S, layer2C)
                            observed = sigmoid(z)
                            currError = currError / observed
                            EI = alpha * (observed - value)
                            summation += EI * layer2C
                            self.layer2[classifierIndex] = layer2C - EI * layer1S

                    for classifierWord, value in negativeClassifiers:
                        if classifierWord != 'UNKNOWN':
                            classifierIndex = self.vocab.vocabToIndex[classifierWord]
                            layer2C = self.layer2[classifierIndex]
                            z = np.dot(layer1S, layer2C)
                            observed = sigmoid(z)
                            currError = currError * observed
                            EI = alpha * (observed - value)
                            summation += EI * layer2C
                            self.layer2[classifierIndex] = layer2C - EI * layer1S

                    currCost += math.log(currError + 1e-9)
                    self.layer1[centerIndex] = layer1S - summation
            alpha = alpha / 2
            # print("current epoch :", i)
            # print("Current Loss : ", currCost)
            # fileName = "vectors" + str(i) + ".txt"
            # outputFile = "./outputs/" + fileName
            # saveFile(self.vocab.vocabWords, self.layer1, outputFile)


