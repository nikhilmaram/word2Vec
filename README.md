# word2Vec
word2Vec and Glove implementation using numpy.

In word2vec and glove, we generate an embedding space for the words.
The program outputs a vector.txt file which contains the embedding vectors.


run_word2vec : runs the word2vec file
run_glove    : runs the glove file.

The run file (run_word2vec, run_glove) assumes the following inputs:
1. text file(text8) : which contains the corpus
2. Dictionary file(vocab.txt) : which contains the dictionary words
3. vectors file(vectors.txt) : An output file, which contain the output embedding vectors.

The dictionary file used for this is included.

The text file can be downloaded from http://mattmahoney.net/ dc/text8.zip . 
The description of the test data set can be found here http://mattmahoney.net/dc/textdata

To run the word2vec :
python run_word2vec

To runt the glove :
python run_glove

The program expects text8 and vocab.txt files in the same directory.
