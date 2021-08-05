import numpy as np 
from keras.models import Model
from keras.layers import Dense,Input,Dropout,LSTM,Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from numpy import random
np.random.seed(1234)
from utils import glove_vector_loader
from utils import *



def sentence_to_indices(X,word_index,max_length):
    number_of_samples = X.shape[0]

    X_indices = np.zeros((number_of_samples,max_length))

    for i in range(number_of_samples):
        sentence_to_words = X[i].lower().split()
        j =0
        for word in sentence_to_words:
            X_indices[i,j] = word_index[word]
            j+=1

    return X_indices

def main():
    word_index,index_word,map = glove_vector_loader('/Users/anirudhsharma/Documents/NLP/Data/glove.6B.50d.txt')
    X1 = np.array(["that is really funny lol", "lets play a game of poker", "the car is waiting for me "])
    X1_indices = sentence_to_indices(X1,word_index, max_length = 7)
    print("X1 =", X1)
    print("X1_indices =", X1_indices)

#git
if __name__ == '__main__':
    main()

