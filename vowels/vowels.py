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

def cov_matrix(dataset): #CROSS CHECK THAT THIS IS CORRECT!!
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

def generate_mean_cov_map(filename,start,end):
    mean_cov_map = {}
    classes_map = ext.extract_classes_map(filename)
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
        train_map[sound] = classes_map[sound][start:end]
        test_map[sound] = classes_map[sound][end:]

    return train_map,test_map

def single_gaussian(start,end,diag = False):

    
    mean_cov_map,sound_list = generate_mean_cov_map("data.dat",start,end)
    train_map,test_map = generate_x("data.dat",start,end)
    probability_vector = np.empty((12,1))
    correct =  0
    wrong = 0
    total = 0
    confusion_matrix = np.zeros((12,12))
    true_index = 0
    for iterate_class in train_map:
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
                confusion_matrix[predicted_index][true_index] += 1
        true_index += 1
    print(confusion_matrix)
    print(correct/total)






print(single_gaussian(0,70,True))

