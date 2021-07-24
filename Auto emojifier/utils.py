import csv
import numpy as np 
import pandas as pd 
import emoji 
import seaborn as sns 
import matplotlib.pyplot as plt
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
        Y=np.array(emoticon).astype(int)
    return X,Y



def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

emoji_dictionary = {"0": ":blue_heart:",   
                    "1": ":football:",
                    "2": ":grin:",
                    "3": ":disappointed:",
                    "4": ":fork:"}


def change_to_emoji(text):
    return emoji.emojize(emoji_dictionary[str(text)],use_aliases=True)

def predict(X,Y,weight,bias,map):
    m = X.shape[0]
    prediction = np.zeros((m,1))

    for i in range(m):
        words = X[i].lower().split()
        average = np.zeros((50,1))

        for word in words:
            average += map[word]
        average = average/len(words)


        #forward pass
        Z = np.dot(weight,average) + bias
        A = softmax_function(Z)
        prediction[i]= np.argmax(A)

    print("Accuracy ----->" + str(np.mean((prediction[:] == Y.reshape(Y.shape[0],1)[:]))))

    return prediction

def display_predictions(X,prediction):
    for i in range(X.shape[0]):
        print(X[i],change_to_emoji(int(prediction[i])))



def confusion_matrix(y_original,y_predicted):
    title = 'Confusion Matrix'
    cmap= plt.cm.gray_r

    confusion_matrix_df = pd.crosstab(y_original,y_predicted.reshape(y_predicted.shape[0],)
                                        ,rownames=['Actual'],colnames=['Predicted'],margins=True)
    
    normalised_matrix = confusion_matrix_df/confusion_matrix_df.sum(axis=1)

    plt.matshow(confusion_matrix_df,cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(confusion_matrix_df.columns))
    plt.xticks(tick_marks,confusion_matrix_df.columns,rotation=45)
    plt.yticks(tick_marks,confusion_matrix_df.index)
    plt.ylabel(confusion_matrix_df.index.name)
    plt.xlabel(confusion_matrix_df.index.name)






            

