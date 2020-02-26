#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python

import json
import numpy as np

def text_spliter(text):
    return text.lower().split(" ")

def getWordVectors(data_set,top_words):
    ''' Gets word vectors + avg char/word '''
    for data_point in data_set:
        data_point['word_vector'] = {}
        split_text = text_spliter(data_point['text'])
        data_point['big_words'] = 0
        data_point['is_parent'] = 0
        char_count = 0
        for word in split_text:
            char_count = char_count + len(word)
            if (word in top_words):
                index = top_words.index(word)
                if (index in data_point['word_vector']):
                    data_point['word_vector'][index] = data_point['word_vector'][index] + 1
                else:
                    data_point['word_vector'][index] = 1
            if (len(word) >= 5):
                data_point['big_words'] = data_point['big_words'] + 1
        if (data_point['is_root'] == True) & (data_point['children'] > 0):
            data_point['is_parent'] = 1

        #data_point['avg_char_word'] = char_count / word_count
    return data_set

def getMatrices(data_set,x_matrix,y_matrix):
    count = 0
    for data_point in data_set:
        if (data_point['is_root']):
            x_matrix[count,0] = 1
        else:
            x_matrix[count,0] = 0

        x_matrix[count,1] = data_point['controversiality']
        x_matrix[count,2] = data_point['children']
        x_matrix[count,3] = data_point['big_words']
        x_matrix[count,4] = 1 if (len(data_point['text']) > 50) else 0
        x_matrix[count,5] = data_point['is_parent']

        for index, wc in data_point['word_vector'].items():
            x_matrix[count,6+index] = wc

        y_matrix[count] = data_point['popularity_score']
        count = count + 1

    return x_matrix,y_matrix

def closedForm(data_set, nb_top_words):
    x_matrix = np.zeros((len(data_set),nb_top_words + 7))
    x_matrix[:,nb_top_words+6] = 1
    y_matrix = np.zeros((len(data_set),1))
    x_matrix,y_matrix = getMatrices(training_set,x_matrix,y_matrix)

    x_matrix_transpose = np.matrix.transpose(x_matrix)

    return np.matmul(np.linalg.inv(np.matmul(x_matrix_transpose,x_matrix)),np.matmul(x_matrix_transpose,y_matrix))

def gradientDescent(data_set):
    initial_learning_rate = 0.0000000001
    decay_constant = 0.001
    epsilon = 0.0001

    x_matrix = np.zeros((len(data_set),nb_top_words + 7))
    x_matrix[:,nb_top_words+6] = 1
    y_matrix = np.zeros((len(data_set),1))

    weight_vector = np.ones((nb_top_words + 7,1))

    x_matrix, y_matrix = getMatrices(data_set,x_matrix,y_matrix)
    x_matrix_transpose = np.matrix.transpose(x_matrix)
    xtx_matrix = np.matmul(x_matrix_transpose,x_matrix)
    xty_matrix = np.matmul(x_matrix_transpose,y_matrix)
    iteration_count = 1
    while True:
        learning_rate = initial_learning_rate / (1 + decay_constant * iteration_count)
        xtxw_xty_matrix = 2*learning_rate*(np.subtract(np.matmul(xtx_matrix,weight_vector),xty_matrix))

        weight_vector_new = np.subtract(weight_vector,xtxw_xty_matrix)
        if (np.linalg.norm(np.subtract(weight_vector_new,weight_vector)) < epsilon):
            weight_vector = weight_vector_new
            break
        weight_vector = weight_vector_new
        iteration_count = iteration_count + 1
    print(iteration_count)
    return weight_vector

def errorCalc(w_vector,data_set):
    error = 0
    for data_point in data_set:
        if (data_point['is_root']):
            vector_sum = w_vector[0]*1
        else:
            vector_sum = 0
        vector_sum = w_vector[1]*data_point['controversiality']
        vector_sum = w_vector[2]*data_point['children']
        vector_sum = w_vector[3]*data_point['big_words']
        vector_sum = w_vector[4]*(1  if (len(data_point['text']) > 50) else 0)
        vector_sum = w_vector[5]*data_point['is_parent']

        for index, value in data_point['word_vector'].items():
            vector_sum = w_vector[index+6]*value
        error = error + (data_point['popularity_score'] - vector_sum) ** 2

    return (error/len(data_set))**0.5


''' Start of actual script (main) '''
with open("proj1_data.json") as fp:
    data = json.load(fp)

nb_top_words = 160

training_set = data[0:9999]
validation_set = data[10000:10999]
test_set = data[11000:11999]

word_counts = {}
for data_point in training_set:
    split_text = text_spliter(data_point['text'])
    for word in split_text:
        if (word in word_counts):
            word_counts[word] = word_counts[word] + 1
        else:
            word_counts[word] = 1

sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
top_words = sorted_words[0:nb_top_words]

training_set = getWordVectors(training_set,top_words)

closed_weight_matrix = closedForm(training_set,nb_top_words)

validation_set = getWordVectors(validation_set,top_words)



print (errorCalc(closed_weight_matrix,validation_set))

print (errorCalc(closed_weight_matrix,training_set))

gd_weight_matrix = gradientDescent(training_set)

print (errorCalc(gd_weight_matrix, validation_set))

print (errorCalc(gd_weight_matrix, training_set))
