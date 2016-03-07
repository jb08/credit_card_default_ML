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
    data_sets = pd.read_csv(file_name, index_col = 0, header = 0, skiprows = [1], usecols= range(24))
    labels = pd.read_csv(file_name, index_col = 0, header = 0, skiprows = [1], usecols= [0,24])

    #data_sets = data_sets.as_matrix()

    return data_sets, labels
    # data_sets = data_sets.as_matrix() # how to convert to ndarray


def main():
    data_sets, labels = csv_reader()
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


    # KNN_classifer = build_KNN_classifier(data_sets, labels)
    # KNN_predicted = predict_test_data(data_sets, KNN_classifer)
    # ROC.calc_confusion_matrix("KNN", KNN_predicted, labels)

    # data_sets = data_sets.as_matrix()
    # knn_probas = KNN_classifer.predict_proba(data_sets)
    
    # ROC.build_roc_curve(labels, knn_probas)
    folds = 10

    # for pca_n in range(1, 23):
    #     print "pca n = ", pca_n, "-----------------------------------"
    #     NB_cross_validation(data, folds, pca_n)

    # LR_cross_validation(data, folds)
    # DA_cross_validation(data, folds)
    # NB_cross_validation(data, folds)
    # DT_cross_validation(data, folds)
    SVM_cross_validation(data, folds)


if __name__ == "__main__":
    main()