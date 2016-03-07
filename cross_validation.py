import sys
import csv
import numpy as np
from classifier import *
from sklearn.cross_validation import KFold

def KNN_cross_validation(data_sets, folds):
    kf = KFold(30000, n_folds=folds)
    final_error = 0.0
    index = 1
    for train_indices, test_indices in kf:
        X_train = []
        X_test = []
        for i in train_indices:
            X_train.append(data_sets[i])
        for i in test_indices:
            X_test.append(data_sets[i])
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for row in X_train:
            train_x.append(row[0:22])
            train_y.append(row[23])
        for row in X_test:
            test_x.append(row[0:22])
            test_y.append(row[23])
        KNN_classifer = build_KNN_classifier(train_x, train_y)
        KNN_predicted = predict_test_data(test_x, KNN_classifer)
        KNN_error_rate = error_measure(KNN_predicted, test_y)
        print index, " fold KNN_error_rate: ",  KNN_error_rate
        final_error += KNN_error_rate
        index = index + 1
    print "final_error: ", final_error / float(folds)


def LR_cross_validation(data_sets, folds):
    kf = KFold(30000, n_folds=folds)
    final_error = 0.0
    index = 1
    for train_indices, test_indices in kf:
        X_train = []
        X_test = []
        for i in train_indices:
            X_train.append(data_sets[i])
        for i in test_indices:
            X_test.append(data_sets[i])
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for row in X_train:
            train_x.append(row[0:22])
            train_y.append(row[23])
        for row in X_test:
            test_x.append(row[0:22])
            test_y.append(row[23])
        LR_classifer = build_LR_classifier(train_x, train_y)
        LR_predicted = predict_test_data(test_x, LR_classifer)
        LR_error_rate = error_measure(LR_predicted, test_y)
        print index, " fold LR_error_rate: ",  LR_error_rate
        final_error += LR_error_rate
        index = index + 1
    print "final_error: ", final_error / float(folds)


def DA_cross_validation(data_sets, folds):
    kf = KFold(30000, n_folds=folds)
    final_error = 0.0
    index = 1
    for train_indices, test_indices in kf:
        X_train = []
        X_test = []
        for i in train_indices:
            X_train.append(data_sets[i])
        for i in test_indices:
            X_test.append(data_sets[i])
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for row in X_train:
            train_x.append(row[0:22])
            train_y.append(row[23])
        for row in X_test:
            test_x.append(row[0:22])
            test_y.append(row[23])
        DA_classifer = build_DA_classifier(train_x, train_y)
        DA_predicted = predict_test_data(test_x, DA_classifer)
        DA_error_rate = error_measure(DA_predicted, test_y)
        print index, " fold DA_error_rate: ",  DA_error_rate
        final_error += DA_error_rate
        index = index + 1
    print "final_error: ", final_error / float(folds)


def DT_cross_validation(data_sets, folds):
    kf = KFold(30000, n_folds=folds)
    final_error = 0.0
    index = 1
    for train_indices, test_indices in kf:
        X_train = []
        X_test = []
        for i in train_indices:
            X_train.append(data_sets[i])
        for i in test_indices:
            X_test.append(data_sets[i])
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for row in X_train:
            train_x.append(row[0:22])
            train_y.append(row[23])
        for row in X_test:
            test_x.append(row[0:22])
            test_y.append(row[23])
        DT_classifer = build_DT_classifier(train_x, train_y)
        DT_predicted = predict_test_data(test_x, DT_classifer)
        DT_error_rate = error_measure(DT_predicted, test_y)
        print index, " fold DT_error_rate: ",  DT_error_rate
        final_error += DT_error_rate
        index = index + 1
    print "final_error: ", final_error / float(folds)


def NB_cross_validation(data_sets, folds):
    kf = KFold(30000, n_folds=folds)
    final_error = 0.0
    index = 1
    for train_indices, test_indices in kf:
        X_train = []
        X_test = []
        for i in train_indices:
            X_train.append(data_sets[i])
        for i in test_indices:
            X_test.append(data_sets[i])
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for row in X_train:
            train_x.append(row[0:22])
            train_y.append(row[23])
        for row in X_test:
            test_x.append(row[0:22])
            test_y.append(row[23])
        NB_classifer = build_NB_classifier(train_x, train_y)
        NB_predicted = predict_test_data(test_x, NB_classifer)
        NB_error_rate = error_measure(NB_predicted, test_y)
        print index, " fold NB_error_rate: ",  NB_error_rate
        final_error += NB_error_rate
        index = index + 1
    print "final_error: ", final_error / float(folds)

