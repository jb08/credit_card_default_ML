from __future__ import division
import sys
import csv
import numpy as np
import pandas as pd
from pandas import DataFrame
from classifier import *
from sklearn.metrics import confusion_matrix, roc_curve, auc
from cross_validation import *
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
import random
import ROC

# author: Jason + Ben

def csv_reader():
    file_name = "default of credit card clients.csv"
    data_sets = pd.read_csv(file_name, index_col = 0, skiprows = [1], header = 0,  usecols= range(24))
    labels = pd.read_csv(file_name, index_col = 0, skiprows = [1], header = 0,  usecols= [0,24])
    return data_sets, labels

def main():
    data_sets, labels = csv_reader()
    ROC.run_analysis(data_sets, labels) 

    #PCA and 10-fold validation
    data, labels2 = csv_reader()
    data["Y"] = labels2
    data = data.as_matrix()

    #print data[0]
    #print "---"
    #print data_sets[0]
    #data = data_sets
    

    # kf = KFold(20, n_folds=folds)
    # for train_indexs, test_indexs in kf:
    #     print train_indexs, test_indexs
    # data_sets, labels = csv_reader()
    # labels = np.ravel(labels)

    folds = 10

    # for pca_n in range(1, 23):
    #     print "pca n = ", pca_n, "-----------------------------------"
    #     NB_cross_validation(data, folds, pca_n)

    # LR_cross_validation(data, folds)
    # DA_cross_validation(data, folds)
    # NB_cross_validation(data, folds)
    # DT_cross_validation(data, folds)
    # LDA_cross_validation(data, folds)
    # SVM_cross_validation(data, folds)


if __name__ == "__main__":
    main()