import numpy as np 
import emoji
from utils import *
import matplotlib.pyplot as plt 

def calculate_avg_sentence(body,map):
    """
    To convert a string in a list of words, extraction of Glove representation of each word 
    and return average value into a single vector. By encoding the sentence meaning.

    Agrs:
        body: (String) training sample for Xtrain.
        map: (Dict) Dictionary map of every word in vocabulary into 50 dim vevtor representation.
    Returns:
        average:(np.array (50,)) Avervage vector encoding info about body.

    Steps:
        1. Split sentence into lower case words
        2. Average the word sentence, loop over words in list
    """
    word_entities = body.lower().split()
    average = np.zeros((50, ))

    for word in word_entities:
        average += map[word]
    average /= len(word_entities)

    return average

def baseline_model(X,Y,map,lr=0.01,iter=500):
    
    return prediction, weight, bais


def main():
    Xtrain,Ytrain = csv_loader('/Users/anirudhsharma/Documents/NLP/Data/train.csv')
    Xtest,Ytest = csv_loader('/Users/anirudhsharma/Documents/NLP/Data/test.csv')
    # print(Xtrain.head(10))
    print(Ytrain)
    maxLen = len(max(Xtrain,key=len).split())

    ''' To test emoji's corresponting to texts '''
    # for i in range(0,len(Xtrain)):
    #     print(Xtrain[i],change_to_emoji(Ytrain[i]))

    '''The output will be a probability vector of shape (1,5), that you then pass in an argmax 
    layer to extract the index of the most likely emoji output.'''

    # Ytrain_oh = convert_to_one_hot(Ytrain,C=5)
    # Ytest_oh = convert_to_one_hot(Ytest,C=5)
    ''' To test training set's corresponting one hote conversion '''

    Ytrain_oh = []

    for i in range(0,len(Ytrain)):
        oh =convert_to_one_hot(Ytrain[i],5)
        Ytrain_oh.append(oh)

    index = 50
    print(Ytrain[index], "is converted into one hot", Ytrain_oh[index])
    print(type(Ytrain))

    Ytest_oh = []

    for i in range(0,len(Ytest)):
        oh1 =convert_to_one_hot(Ytest[i],5)
        Ytest_oh.append(oh1)

    index = 10
    print(Ytest[index], "is converted into one hot", Ytest_oh[index])
    print(type(Ytest_oh))

    word_index,index_word,map = glove_vector_loader('/Users/anirudhsharma/Documents/NLP/Data/glove.6B.50d.txt')
    



    
baseline()