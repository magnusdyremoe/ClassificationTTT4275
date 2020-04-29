import numpy as np
import extract_classes as ext
from scipy.stats import multivariate_normal


def sample_mean(dataset):
    sample_size = len(dataset)
    N = len(dataset[0])
    x_sum = [0]*N
    for vector in range (sample_size):
        for element in range (N):
            x_sum[element] += (int(dataset[vector][element]))
    for element in range(len(x_sum)):
        x_sum[element] /= sample_size
        
    return x_sum

def cov_matrix(dataset): 
    N = len(dataset[0])
    sample_size = len(dataset)
    mean = sample_mean(dataset)
    cov_matrix = np.zeros((N,N))
    x = np.asfarray(dataset, float)
    cov_matrix = np.dot((x-mean).T,(x-mean))/(sample_size-1)  
    return cov_matrix

def make_sequence(sounds):
    list_of_sounds= []
    for sound in sounds:
        list_of_sounds.append(sound)
    return list_of_sounds

def generate_mean_cov_map(classes_map,start,end):
    mean_cov_map = {}
    list_of_sounds = make_sequence(classes_map)
    for sound in classes_map:
        mean_cov_map[sound] =[sample_mean(classes_map[sound][start:end])]
        mean_cov_map[sound].append(cov_matrix(classes_map[sound][start:end]))
    return mean_cov_map,list_of_sounds


def generate_x(filename,start,end):
    test_map = {}
    train_map={}
    classes_map = ext.extract_classes_map(filename)
    for sound in classes_map:
        train,test = equal_representation(classes_map[sound])
        test_map[sound] = test
        train_map[sound] = train

    return train_map,test_map

def train_test_single_gaussian(start,end,diag = False):
    train_map,test_map = generate_x("data.dat",start,end)
    mean_cov_map,sound_list = generate_mean_cov_map(train_map,start,end)

    #---------------------Training------------------------------
    probability_vector = np.empty((12,1))
    correct =  0
    wrong = 0
    total = 0
    confusion_matrix_train = np.zeros((12,12))
    true_index = 0
    for iterate_class in train_map:
        for sample in range(len(train_map[iterate_class])):
            for i,sound in enumerate(mean_cov_map):
                cov_matrix = mean_cov_map[sound][1]
                sample_mean = mean_cov_map[sound][0]
               
                if diag == True:
                    cov_matrix = np.diag(np.diag(mean_cov_map[sound][1]))
                    probability = multivariate_normal.pdf(train_map[iterate_class][sample],mean = sample_mean,cov = cov_matrix)
                else:
                    probability = multivariate_normal.pdf(train_map[iterate_class][sample],mean = sample_mean,cov = cov_matrix,allow_singular = True)
                probability_vector[i] = probability
            predicted_index = np.argmax(probability_vector)
            predicted_sound = sound_list[predicted_index]
            true_guess = iterate_class
            total += 1
            if true_guess == predicted_sound:
                correct += 1
            else:
                wrong += 1
            confusion_matrix_train[true_index][predicted_index] += 1
        true_index += 1
    print("Training : ")
    print(confusion_matrix_train)
    print(correct/total)
    print(total)

    #-----------------------------Testing------------------------------
    probability_vector = np.empty((12,1))
    correct =  0
    wrong = 0
    total = 0
    confusion_matrix_test = np.zeros((12,12))
    true_index = 0

    for iterate_class in test_map:
        for sample in range(len(test_map[iterate_class])):
            for i,sound in enumerate(mean_cov_map):
                cov_matrix = mean_cov_map[sound][1]
                sample_mean = mean_cov_map[sound][0]
               
                if diag == True:
                    cov_matrix = np.diag(np.diag(mean_cov_map[sound][1]))
                    probability = multivariate_normal.pdf(test_map[iterate_class][sample],mean = sample_mean,cov = cov_matrix)
                else:
                    probability = multivariate_normal.pdf(test_map[iterate_class][sample],mean = sample_mean,cov = cov_matrix,allow_singular = True)
                probability_vector[i] = probability
            predicted_index = np.argmax(probability_vector)
            predicted_sound = sound_list[predicted_index]
            true_guess = iterate_class
            total += 1
            if true_guess == predicted_sound:
                correct += 1
            else:
                wrong += 1
            confusion_matrix_test[true_index][predicted_index] += 1
        true_index += 1
    print("Testing : ")
    print(confusion_matrix_test)
    print(correct/total)
    print(total)
    return confusion_matrix_train, confusion_matrix_test

def equal_representation(dataset):
    test_set = []
    training_set = []
    for man in range(0,20): #20
        training_set.append(dataset[man])
    for woman in range(46,66): #20
        training_set.append(dataset[woman])
    for boy in range(93,113): #20
        training_set.append(dataset[boy])
    for girl in range(120,130): #10
        training_set.append(dataset[girl])
    for man in range(20,46): #26
        test_set.append(dataset[man])
    for woman in range(66,93): #27
        test_set.append(dataset[woman])
    for boy in range(113,120): #7
        test_set.append(dataset[boy])
    for girl in range(130,139): #9
        test_set.append(dataset[girl])
    return training_set,test_set

