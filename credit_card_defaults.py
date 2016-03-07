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
    KNN_cross_validation(data, folds)
    LR_cross_validation(data, folds)
    DA_cross_validation(data, folds)
    NB_cross_validation(data, folds)
    DT_cross_validation(data, folds)

    # KNN_classifer = build_KNN_classifier(data_sets, labels)
    # KNN_predicted = predict_test_data(data_sets, KNN_classifer)
    # KNN_error_rate = error_measure(KNN_predicted, labels)
    # print "KNN_error_rate: ",  KNN_error_rate

    # LR_classifer = build_LR_classifier(data_sets, labels)
    # LR_predicted = predict_test_data(data_sets, LR_classifer)
    # LR_error_rate = error_measure(LR_predicted, labels)
    # print "LR_error_rate: ",  LR_error_rate

    # DA_classifer = build_DA_classifier(data_sets, labels)
    # DA_predicted = predict_test_data(data_sets, DA_classifer)
    # DA_error_rate = error_measure(DA_predicted, labels)
    # print "DA_error_rate: ",  DA_error_rate

    # NB_classifer = build_NB_classifier(data_sets, labels)
    # NB_predicted = predict_test_data(data_sets, NB_classifer)
    # NB_error_rate = error_measure(NB_predicted, labels)
    # print "NB_error_rate: ",  NB_error_rate

    # NN_classifer = build_NN_classifier(data_sets, labels)
    # NN_predicted = predict_test_data(data_sets, NN_classifer)
    # NN_error_rate = error_measure(NN_predicted, labels)
    # print "NN_error_rate: ",  NN_error_rate

if __name__ == "__main__":
    main()