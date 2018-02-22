import numpy as np
import math



def saveFile(vocabWords,layer,fileName):
    f = open(fileName,'w')
    for i in range(len(vocabWords)):
        s = ""
        s = str(vocabWords[i])
        for j in range(300):
            s = s + " " + str(layer[i][j])
        s = s +"\n"
        f.write(s)

class Vocabulary:
    def __init__(self, vocabFile):
        self.vocabFile = vocabFile
        self.vocabWords = []
        self.corpusWordCount = {}
        self.vocabToIndex = {}

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

        return self.vocabWords, self.vocabToIndex

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
                ## Adds one to existing count, if word not present then initialised to zero
                self.corpusWordCount[word] = self.corpusWordCount[word] + 1
            else:
                self.corpusWords[index] = 'UNKNOWN'
        return self.corpusWords

def buildCoOcuuerenceMatrix(vocabWords, vocabIndex, corpusWords, contextSize=10, minCount=None):
    ## Create a matrix of size equal to vocabWordSize * vocabWordSize
    vocabDimension = len(vocabWords)
    coOccurenceMatrix = np.zeros(shape=(vocabDimension, vocabDimension))


    for centerIndex, centerWord in enumerate(corpusWords[contextSize:]):
        ## Get the elements in the left context window and do it for every element, In this way you will not visit the node
        ## multiple times
        if (centerWord != 'UNKNOWN'):
            centerIndexVocab = vocabIndex[centerWord]

            ####leftIndex = min(0,centerIndex - contextSize)

            ## Words that are present in the left context Window
            currContextWords = corpusWords[centerIndex - contextSize:centerIndex]
            ####currContextWindowSize = centerIndex - leftIndex

            currContextWindowSize = contextSize
            ## for each word in the context window update the corresponding cooccurence entry with inverse of distance
            ## Inverse of distance because farther away from the center, lesser the influence
            for index, word in enumerate(currContextWords):
                if (word != 'UNKNOWN'):
                    ## Distance from the center word
                    distance = currContextWindowSize - index
                    ## Get the index of the corresponding word index
                    contextIndexVocab = vocabIndex[word]

                    value = 1 / float(distance)
                    coOccurenceMatrix[centerIndexVocab][contextIndexVocab] += value
                    coOccurenceMatrix[contextIndexVocab][centerIndexVocab] += value

    return coOccurenceMatrix

def preComputeWeights(coOccurenceMatrix,vocabDimension,maxCoOccur=100):
    weightMatrix = np.zeros(shape=(vocabDimension,vocabDimension))
    for i in range(vocabDimension):
        for j in range(vocabDimension):
            weightMatrix[i,j] = ((coOccurenceMatrix[i,j]/maxCoOccur) ** 0.75) if coOccurenceMatrix[i,j] < maxCoOccur else 1
    return weightMatrix

def expectedOutput(coOccurenceMatrix,vocabDimension,minCount=5):
    expectedMatrix = np.zeros(shape=(vocabDimension,vocabDimension))
    for i in range(vocabDimension):
        for j in range(vocabDimension):
            if(coOccurenceMatrix[i][j] > minCount):
                expectedMatrix[i,j] = np.log(coOccurenceMatrix[i][j])
    return expectedMatrix

## In glove the product of vector of center word and context word should match the number of cooccurences of both words.
## In word2Vec we predict the vector of word based on its context
def glove(vocabDimension, coOccurenceMatrix, weightMatrix, expectedMatrix, embedDimension=300, minCount=5, numIter=100, learningRate=0.01):
    np.random.seed(1234)
    mainWeightMatrix = np.random.uniform(low=-0.5 / embedDimension, high=0.5 / embedDimension,
                                         size=(vocabDimension, embedDimension))
    np.random.seed(5678)
    contextWeightMatrix = np.random.uniform(low=-0.5 / embedDimension, high=0.5 / embedDimension,
                                            size=(vocabDimension, embedDimension))
    np.random.seed(8734)
    mainBias = np.random.uniform(low=-0.5 / embedDimension, high=0.5 / embedDimension, size=(vocabDimension))
    np.random.seed(7986)
    contextBias = np.random.uniform(low=-0.5 / embedDimension, high=0.5 / embedDimension, size=(vocabDimension))

    mainWeightMatrixGradSquare = np.ones((vocabDimension, embedDimension))
    contextWeightMatrixGradSquare = np.ones((vocabDimension, embedDimension))
    mainBiasGradSquare = np.ones(vocabDimension)
    contextBiasGradSquare = np.ones(vocabDimension)

    ## Perform Stochastic gradient descent
    indexes = np.random.choice(vocabDimension * vocabDimension, vocabDimension * vocabDimension, replace=False)
    for itera in range(numIter):
        #         for mainIndex in vocabIter:
        #         ##for mainIndex in range(vocabDimension):
        #             np.random.seed(mainIndex)
        #             contextIter = np.random.choice(vocabDimension,vocabDimension,replace=False)
        #             for contextIndex in contextIter:
        #             ##for contextIndex in range(vocabDimension):
        currCost = 0
        for index in indexes:
            mainIndex = int(index / vocabDimension)
            contextIndex = index % vocabDimension

            if coOccurenceMatrix[mainIndex][contextIndex] > minCount:
                mainWeightVector = mainWeightMatrix[mainIndex]
                contextWeightVector = contextWeightMatrix[contextIndex]
                ##print(mainIndex,contextIndex)
                expectedOccurence = expectedMatrix[mainIndex][contextIndex]
                observedOccurence = np.dot(mainWeightVector, contextWeightVector) + mainBias[mainIndex] + contextBias[
                    contextIndex]

                ## The weighted f(x_ij) in weighted regression model
                ###weight = ((expectedOccurence/maxCoOccur) ** 0.75) if expectedOccurence < maxCoOccur else 1
                ## Difference in observed and expected
                diffOccurence = observedOccurence - expectedOccurence

                ## Update the weights using SGD
                ## Updating the word weight vectors
                ##print(weight ,np.dot(mainWeightVector,contextWeightVector) , mainBias[mainIndex] , contextBias[contextIndex], expectedOccurence)
                ##print(diffOccurence, math.log(expectedOccurence))
                # print(weight)
                commonProduct = learningRate * weightMatrix[mainIndex][contextIndex] * diffOccurence

                mainWeightMatrixGrad = commonProduct * contextWeightVector
                contextWeightMatrixGrad = commonProduct * mainWeightVector

                mainWeightMatrix[mainIndex] -= mainWeightMatrixGrad / np.sqrt(mainWeightMatrixGradSquare[mainIndex])
                contextWeightMatrix[contextIndex] -= contextWeightMatrixGrad / np.sqrt(
                    contextWeightMatrixGradSquare[contextIndex])

                ## Updating the bias vectors
                mainBias[mainIndex] -= commonProduct / np.sqrt(mainBiasGradSquare[mainIndex])
                contextBias[contextIndex] -= commonProduct / np.sqrt(contextBiasGradSquare[contextIndex])

                mainWeightMatrixGradSquare[mainIndex] += np.square(mainWeightMatrixGrad)
                contextWeightMatrixGradSquare[contextIndex] += np.square(contextWeightMatrixGrad)
                mainBiasGradSquare[mainIndex] += np.square(commonProduct)
                contextBiasGradSquare[contextIndex] += np.square(commonProduct)

                currCost += weightMatrix[mainIndex][contextIndex] * (diffOccurence ** 2)

        ##print("Epoch ", itera)
        ##print("Current Cost:", currCost)

    return mainWeightMatrix, contextWeightMatrix