import csv
import numpy as np 
import pandas as pd 
import emoji 
import seaborn as sns 
from sklearn.metrics import confusion_matrix

def glove_vector_loader(file):
    with open(file, 'r') as workingFile:
        words = set()
        map ={}
        for line in workingFile:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            map[curr_word]= np.array(line[1:],dtype=np.float64)

        i = 0
        word_index = {}
        index_word={}

        for word in sorted(words):
            word_index[word]= i
            index_word[i]= word
    return word_index,index_word,map



def softmax_function(x):
    """For computing softmax values for each score in x"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def csv_loader(file_name):
    text = []
    emoticon =[]

    with open(file_name,'r') as csv_file:
        read_csv = csv.reader(csv_file)

        for row in read_csv:
            text.append(row[0])
            emoticon.append(row[1])

    X= np.array(text)
    Y=np.array(emoticon)
    return X.Y



def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

