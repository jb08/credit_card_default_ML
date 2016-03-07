from __future__ import division
import sys
import csv
import numpy as np
import pandas as pd
from pandas import DataFrame
from classifier import *
<<<<<<< HEAD
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import random

def csv_reader():
	file_name = "default of credit card clients.csv"
	data_sets = pd.read_csv(file_name, index_col = 0, header = 0, skiprows = [1], usecols= range(24))
	labels = pd.read_csv(file_name, index_col = 0, header = 0, skiprows = [1], usecols= [0,24])

	data_sets = data_sets.as_matrix()

	return data_sets, labels
	# data_sets = data_sets.as_matrix() # how to convert to ndarray

def cardinality(labels):
	yes = 0
	no = 0

	for label in labels:
		if (label == 0):
			no += 1
		else:
			yes += 1

	print "total yes: " + str(yes)
	print "total no: " + str(no)
	print "n: " + str(yes + no)

	percent_default = yes / (yes+no)
	print "percentage defaults: " + str(percent_default)

def build_roc_curve(labels, knn_probas):

	knn_fpr, knn_tpr, knn_thresholds = roc_curve(labels, knn_probas[:, 1])
	roc_auc = auc(knn_fpr, knn_tpr)

	#setup plot
	plt.plot(knn_fpr, knn_tpr, label='KNN classifier')
	
	label=('KNN AUC = %0.2f'% roc_auc)
	#print label

	plt.legend(loc='upper right')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.show()

def calc_confusion_matrix(model, predicted, labels):
	error_rate = error_measure(predicted, labels)
	print model + " error rate: ", error_rate

	cm = confusion_matrix(labels, predicted, labels = [0,1])

	print model + " confusion_matrix: "
	print cm
	print "---"

def main():
	data_sets, labels = csv_reader()
	labels = np.ravel(labels)

	KNN_classifer = build_KNN_classifier(data_sets, labels)
	KNN_predicted = predict_test_data(data_sets, KNN_classifer)
	calc_confusion_matrix("KNN", KNN_predicted, labels)

	data_sets = data_sets.as_matrix()
	knn_probas = KNN_classifer.predict_proba(data_sets)
	
	build_roc_curve(labels, knn_probas)
	

	
	# print false_positive_rate
	# print true_positive_rate
	# print thresholds
	

	# plt.title('Receiver Operating Characteristic')
	# plt.plot(false_positive_rate, true_positive_rate, 'b',

	# plt.plot([0,1],[0,1],'r--')
	# plt.xlim([-0.1,1.2])
	# plt.ylim([-0.1,1.2])
	# plt.ylabel('True Positive Rate')
	# plt.xlabel('False Positive Rate')
	# plt.show()



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
=======
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
>>>>>>> 06e4a5cc35c69f38611e8e8dfab516a62d378506

if __name__ == "__main__":
    main()