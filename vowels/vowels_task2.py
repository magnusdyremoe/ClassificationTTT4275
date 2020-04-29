import numpy as np
import extract_classes as ext
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture as GMM
import vowels_task1 as v


def map_join_array(type_map):
    x = []
    for sound in type_map:
        x.extend(type_map[sound[0:70]])
    return np.asfarray(x)
def generate_sound_list(train_map):
    sounds = []
    for sound in train_map:
        sounds.append(sound)
    return sounds

def train_test_GMM(start,end, n_components):
    train_map,test_map = v.generate_x("data.dat",0,70)
    sound_list = generate_sound_list(train_map)
    

    x_train = map_join_array(train_map)
    x_test = map_join_array(test_map)
    print("Training GMM")
    probabilities_train = np.zeros((12,x_train.shape[0]))
    probabilities_test = np.zeros((12,x_test.shape[0]))
    

    for i,sound in enumerate(train_map):
        gmm = GMM(n_components=n_components, covariance_type='diag', reg_covar=1e-4, random_state=0)
        gmm.fit(train_map[sound], sound_list)
        for j in range(n_components):
            N = multivariate_normal(mean=gmm.means_[j], cov=gmm.covariances_[j], allow_singular=True)
            probabilities_train[i] += gmm.weights_[j] * N.pdf(x_train)
            probabilities_test[i] += gmm.weights_[j] * N.pdf(x_test)

    
    confusion_matrix_train = np.zeros((12,12))
    confusion_matrix_test = np.zeros((12,12))
    
    predict_test = np.argmax(probabilities_test,axis = 0)
    predict_train = np.argmax(probabilities_train,axis = 0)
    true_test = np.asarray([i for i in range (12) for _ in range(70)])
    true_train = np.asarray([i for i in range(12) for _ in range(70)])
    correct =  0
    wrong = 0
    total = 0
    for index in range(len(predict_test)):
        if int(predict_test[index]) == true_test[index]:
                correct += 1
        else:
            wrong += 1
        confusion_matrix_test[true_test[index]][int(predict_test[index])] += 1
        total += 1
    ratio_test = correct/total
    correct = 0
    wrong = 0
    total = 0
    print("Train: ")
    print(confusion_matrix_test)
    print("Testing ratio:",ratio_test)
    for index in range(len(predict_train)):
        if int(predict_train[index]) == true_train[index]:
            correct += 1
        else:
            wrong += 1
        confusion_matrix_train[true_train[index]][int(predict_train[index])] += 1
        total += 1
    ratio_training = correct/total
    print(confusion_matrix_train)
    print("training ratio:",ratio_training)


   
    
    return confusion_matrix_train, confusion_matrix_test



"""

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

def train_test_GMM(start,end, n_components):
    mean_cov_map,sound_list = generate_mean_cov_map("data.dat",start,end)
    train_map,test_map = generate_x("data.dat",start,end)
    #---------------------Training------------------------------
    
    correct =  0
    wrong = 0
    total = 0
    confusion_matrix_test = np.zeros((12,12))
    confusion_matrix_train = np.zeros((12,12))

    probability = 0
    print("Training GMM")
    #print(len(train_map["uw"]))
    probability_vectors = np.zeros((12,len(train_map["uw"])))
    predicted_indeces = np.empty((12,len(train_map["uw"])))
    for i,sound in enumerate(train_map):
        x = np.asfarray(train_map[sound], float)
        gmm = GMM(n_components=n_components, covariance_type='diag', reg_covar=1e-4, random_state=0)
        gmm.fit(train_map[sound], sound_list)
        for j in range(n_components):
            N = multivariate_normal(mean=gmm.means_[j], cov=gmm.covariances_[j], allow_singular=True)
            probability += gmm.weights_[j] * N.pdf(x)
        
        
        probability_vectors[i] = probability 
        predicted_indeces[i] = np.argmax(probability_vectors,axis = 0)
    for j,sound in enumerate(predicted_indeces):
        for guess in sound:
            if int(guess) == j:
                    correct += 1
            else:
                wrong += 1
            confusion_matrix_train[j][int(guess)] += 1
            total += 1

    print("Training : ")
    print(confusion_matrix_train)
    #print(confusion_matrix_train)
    print(correct/total)
    return confusion_matrix_train, confusion_matrix_test

    """