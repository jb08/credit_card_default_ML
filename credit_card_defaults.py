import sys
import csv
import numpy as np
import pandas as pd
from pandas import DataFrame
from classifier import *
from sklearn.cross_validation import KFold

def readfile():
    file_name = "PreprocesedData.csv"
    openfile = open(file_name)
    file_data = csv.reader(openfile, delimiter=",")
    X = []
    Y = []
    for row in file_data:
    	X.append(row[0 : 22])
    	Y.append(row[23])
    return X, Y


def ten_fold_cross_validation(X):
    kf = KFold(30000, n_folds=2)
    X_train = da
    # print len(data_sets), len(labels)
    # for train_indexs, test_indexs in kf:

def main():
	X, Y = readfile()
	print Y[29999]
	print X[29999]

    # print ten_fold_cross_validation(data_sets, labels)

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