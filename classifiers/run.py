'''
Created on May 16, 2018

@author: andres
'''

# ---------------------------------------------------------------------------
# Para garantir reprodutibilidade
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
from sklearn import preprocessing
set_random_seed(2)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Importing the libraries
import os
import sys
import numpy as np
import pandas as pd
# ---------------------------------------------------------------------------

# Paramter Verification
#if (len(sys.argv) != 2):
#    print("You need to give the directory path of train and test data.")
#    exit()
    
dir_path = "D:\\Camila Leite\\Movelets Foursquare\\Experimentos\\results\\10_week\\Movelets\\newfour8_ED"
#dir_path = "D:\\Camila Leite\\Movelets Foursquare\\Experimentos\\results\\DASData\\Movelets\\Com QUATRO parametros e PRUNNING"

print("Loading train and test data from... " + dir_path)
dataset_train = pd.read_csv(dir_path + "\\2grams_train.csv")
dataset_test  = pd.read_csv(dir_path + "\\2grams_test.csv")
print("Done.")

nattr = len(dataset_train.iloc[1,:])
print("Number of attributes: " + str(nattr))

# Separating attribute data (X) than class attribute (y)
X_train = dataset_train.iloc[:, 0:(nattr-1)].values
y_train = dataset_train.iloc[:, (nattr-1)].values

X_test = dataset_test.iloc[:, 0:(nattr-1)].values
y_test = dataset_test.iloc[:, (nattr-1)].values

X_train = preprocessing.scale(X_train);   
X_test = preprocessing.scale(X_test);

# Replace distance 0 for presence 1
# and distance 2 to non presence 0
#X_train[X_train == 2] = 0
#X_test[X_test == 0] = 1
#X_test[X_test == 2] = 0

# ----------------------------------------------------------------------------------
from Methods import Approach1, Approach2, ApproachRF, ApproachRFHP , ApproachMLP, ApproachDT

par_droupout = 0.7
par_batch_size = 200
par_epochs = 80
par_lr = 0.00095
save_results = True

# ----------------------------------------------------------------------------------
# Building the neural network-
print("Building neural network")

#Approach1(X_train, y_train, X_test, y_test, par_batch_size, par_epochs, par_lr, par_droupout, save_results, dir_path)

lst_par_epochs = [80]
lst_par_lr = [0.00095,0.00075,0.00055,0.00025,0.00015]
#Approach2(X_train, y_train, X_test, y_test, par_batch_size, lst_par_epochs, lst_par_lr, par_droupout, save_results, dir_path)

print("Done.")

print("Building random forest models")

# Este experimento eh para fazer uma varredura de arvores em random forestx
#n_estimators = np.arange(10, 751, 10)
#n_estimators = np.append([1], n_estimators)
n_estimators = [300]
print(n_estimators)

#Estou usando este de baixo (camila)
ApproachMLP(X_train, y_train, X_test, y_test, par_batch_size, par_epochs, par_lr, par_droupout, save_results, dir_path)

#ApproachRF(X_train, y_train, X_test, y_test, n_estimators, save_results, dir_path)

#ApproachRFHP(X_train, y_train, X_test, y_test, save_results, dir_path)

#ApproachDT(X_train, y_train, X_test, y_test, save_results, dir_path)

#ApproachSVM(X_train, y_train, X_test, y_test, save_results, dir_path)

# ---------------------------------------------------------------------------------
print("Done.")
print("Finished.")