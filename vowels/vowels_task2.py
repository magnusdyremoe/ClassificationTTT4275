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
    print("Testing: ")
    print(confusion_matrix_test)
    print("Testing ratio:",ratio_test)
    print("\n--------------------------------\n")
    for index in range(len(predict_train)):
        if int(predict_train[index]) == true_train[index]:
            correct += 1
        else:
            wrong += 1
        confusion_matrix_train[true_train[index]][int(predict_train[index])] += 1
        total += 1
    ratio_training = correct/total
    print("Training: ")
    print(confusion_matrix_train)
    print("training ratio:",ratio_training)
    
    return confusion_matrix_train, confusion_matrix_test



