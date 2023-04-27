import argparse
#from dataset import DataSplit
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(
    description="standard k-nearest neighbors.")
parser.add_argument("--dataset", type=str)

args = parser.parse_args()
dataset = args.dataset

if dataset == 'mnist':
    (train_images , train_labels) , (test_images , test_labels ) = mnist.load_data()
    flatten = train_images.shape[1] * train_images.shape[2]
    X_train = train_images.reshape(len(train_images), flatten )
    X_test = test_images.reshape(len(test_images), flatten )

    ## Grey Scale is 0 to 255. Divide by 255 to normalize values 0 to 1.
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    
elif dataset == 'cifar10':

    (train_images , train_labels) , (test_images , test_labels ) = cifar10.load_data()
    #flatten = train_images.shape[1] * train_images.shape[2]
    X_train = np.reshape(train_images, (train_images.shape[0], -1))
    X_test = np.reshape(test_images, (test_images.shape[0], -1))

    ## Grey Scale is 0 to 255. Divide by 255 to normalize values 0 to 1.
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    
elif dataset == 'cifar100':

    (train_images , train_labels) , (test_images , test_labels ) = cifar100.load_data()
    #flatten = train_images.shape[1] * train_images.shape[2]
    X_train = np.reshape(train_images, (train_images.shape[0], -1))
    X_test = np.reshape(test_images, (test_images.shape[0], -1))

    ## Grey Scale is 0 to 255. Divide by 255 to normalize values 0 to 1.
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

else:
    raise NotImplementedError()
 
    
    

    # Build KNN model
target_names = np.unique(train_labels)
k_values = [1, 3, 5, 7, 9]
accuracy = []
for k in k_values:
    target_names = np.unique(train_labels)
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(X_train, train_labels.ravel())
    predicted_labels = model.predict(X_test)
    model_accuracy = classification_report(y_true=test_labels.ravel(), y_pred=predicted_labels,  # Get the accuracy for each k
                          target_names=target_names, output_dict=True)['accuracy']
    accuracy.append(model_accuracy)
    
print('the values of accuracy for each values of k', accuracy)
print('the best accuracy is', np.max(accuracy))

 #Plot results
plt.plot(k_values, accuracy)
plt.xlabel('k')
plt.ylabel('model accuracy')
plt.savefig('books_read.png')

