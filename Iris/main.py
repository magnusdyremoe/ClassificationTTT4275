#Iris task 1 and 2

from sklearn.datasets import load_iris
import numpy as np

import iris

iris_data = load_iris()
target = iris_data.target
data = iris_data.data

alpha = 0.01
n_iterations = 3000

print("Target names: {}".format(iris_data['target_names']))

def task_1a():
    training_data1 = data[0:30]
    training_data2 = data[50:80]
    training_data3 = data[100:130]
    training_data = [training_data1,training_data2,training_data3]
    training_target1 = target[0:30]
    training_target2 = target[50:80]
    training_target3 = target[100:130]
    training_target = [training_target1, training_target2, training_target3]

    testing_data1 = data[30:50]
    testing_data2 = data[80:100]
    testing_data3 = data[130:150]
    testing_data = [testing_data1,testing_data2,testing_data3]
    testing_target1 = target[30:50]
    testing_target2 = target[80:100]
    testing_target3 = target[130:150]
    testing_target = [testing_target1, testing_target2, testing_target3]

    #------------------Training-------------------
    training_ratio, confusion_training, W = iris.assignment_1_trainingset(training_data,training_target,3,n_iterations,alpha)
    print("Training:")
    print(training_ratio)
    print(confusion_training)

    #-------------------Testing---------------------
    test_ratio, confusion_test = iris.assignment_1_testingset(W, training_data, testing_data, testing_target, 3,n_iterations,alpha)
    print("Testing:")
    print(test_ratio)
    print(confusion_test)  


def task_1b():
    training_data1 = data[20:50]
    training_data2 = data[70:100]
    training_data3 = data[120:150]
    training_data = [training_data1,training_data2,training_data3]
    training_target1 = target[20:50]
    training_target2 = target[70:100]
    training_target3 = target[120:150]
    training_target = [training_target1, training_target2, training_target3]

    testing_data1 = data[0:20]
    testing_data2 = data[50:70]
    testing_data3 = data[100:120]
    testing_data = [testing_data1,testing_data2,testing_data3]
    testing_target1 = target[0:20]
    testing_target2 = target[50:70]
    testing_target3 = target[100:120]
    testing_target = [testing_target1, testing_target2, testing_target3]

    #------------------Training-------------------

    training_ratio, confusion_training, W = iris.assignment_1_trainingset(training_data,training_target,3,n_iterations,alpha)
    print("Training:")
    print(training_ratio)
    print(confusion_training)

    #-------------------Testing---------------------
    test_ratio, confusion_test = iris.assignment_1_testingset(W, training_data, testing_data, testing_target, 3,n_iterations,alpha)
    print("Testing:")
    print(test_ratio)
    print(confusion_test)  

def task_2(data, target):
    """
    #Historgrams
    features = ['sepal length', 'sepal width', 'petal length', 'petal width']
    iris.printHistograms(data, features)
    """
    #-----------------Removing 1 feature------------
    data = np.delete(data, 1, axis=1)
    training_data, training_target, testing_data, testing_target = iris.allocate_data(data, target)

    print("Training with 3 features: ")
    training_ratio1, confusion_training1, W1 = iris.assignment_1_trainingset(training_data,training_target,3,n_iterations,alpha)
    print(training_ratio1)
    print(confusion_training1)
    print("Testing with 3 features:")
    test_ratio1, confusion_test1 = iris.assignment_1_testingset(W1, training_data, testing_data, testing_target, 3,n_iterations,alpha)
    print(test_ratio1)
    print(confusion_test1) 

    #------------------Removing 2 features-------------
    data = np.delete(data, 0, axis=1)
    training_data, training_target, testing_data, testing_target = iris.allocate_data(data, target)

    print("\n\nTraining with 2 features: ")
    training_ratio2, confusion_training2, W2 = iris.assignment_1_trainingset(training_data,training_target,3,n_iterations,alpha)
    print(training_ratio2)
    print(confusion_training2)
    print("Testing with 2 features:")
    test_ratio2, confusion_test2 = iris.assignment_1_testingset(W2, training_data, testing_data, testing_target, 3,n_iterations,alpha)
    print(test_ratio2)
    print(confusion_test2) 

    #---------------Removing 3 features------------------------------
    data = np.delete(data, 0, axis=1)
    training_data, training_target, testing_data, testing_target = iris.allocate_data(data, target)
    
    print("\n\nTraining with 1 feature: ")
    training_ratio3, confusion_training3, W3 = iris.assignment_1_trainingset(training_data,training_target,3,n_iterations,alhpa)
    print(training_ratio3)
    print(confusion_training3)
    print("Testing with 1 feature:")
    test_ratio3, confusion_test3 = iris.assignment_1_testingset(W3, training_data, testing_data, testing_target, 3,n_iterations,alpha)
    print(test_ratio3)
    print(confusion_test3) 






def main():
    
    print("Task 1a")
    task_1a()
    print("\n ----------------------------------- \n")
    print("Task 1b")
    task_1b()
    
    print("\n ------------------------------------------ \n")
    print("Task 2")
    task_2(data, target)
    
main()