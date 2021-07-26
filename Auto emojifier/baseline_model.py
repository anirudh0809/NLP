import numpy as np 
import emoji
from numpy import random
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
    """  
    Args: 
        X: input, numpy array of text as str (m,1)
        Y: labels (Dependant var)
        map: Dict mapping of every word in a vocabulary into its 50- dimentionsal vector representation 
        lr: Learning rate
        iter: Number of iterations 

    Returns:
        prediction: Predicted vectors (m,1)
        weight: weight matrix of softmax layer 
        bias: bias of the softmax layer 
    """

    np.random.seed(123456)

    training_examples = Y.shape[0]
    num_classes = 5
    glove_dim=50
    weight = np.random.randn(num_classes,glove_dim)/np.sqrt(glove_dim)
    bias = np.zeros((num_classes,))

    Y_one_hot = convert_to_one_hot(Y,C=num_classes)

    #Optimise 
    for i in range(num_classes,glove_dim):
        for j in range(training_examples):
            average = calculate_avg_sentence(X[j],map)
            #forward pass
            z = np.dot(weight,average) + bias
            a = softmax_function(z)

            penalty = -np.sum(np.matmul(Y_one_hot[j],np.log(a)))

            #gradient
            dz = a - Y_one_hot[j]
            dweight = np.dot(dz.reshape(num_classes,1),average.reshape(1,glove_dim))
            dbias = dz

            # parameter Update with SGD
            weight = weight - lr * dweight
            bias - bias - lr * dbias
        if i % 100 == 0 :
            print(f'Epoch : {str(i)} , cost : {str(penalty)}')
            prediction = predict(X,Y,weight,bias,map=map)


    return prediction, weight, bias


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

    # #test 
    # text = "carrot"
    # print(f'index of word {text} is {word_index[text]}')
    # print(index_word[289846])
    pred, W, b = baseline_model(Xtrain, Ytrain, map)
    print(pred)


    



if __name__ == '__main__':
    main()