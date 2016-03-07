from __future__ import division
import sys
import csv
import numpy as np
import pandas as pd
from pandas import DataFrame
from classifier import *
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import random

def run_analysis(data_sets, labels):
	print "ROC::run_analysis()"
	#print_data(data_sets, labels)	

	#pre-process data
	labels = np.ravel(labels)

	KNN_classifer = build_KNN_classifier(data_sets, labels)
	KNN_predicted = predict_test_data(data_sets, KNN_classifer)
	#calc_confusion_matrix("KNN", KNN_predicted, labels)

	knn_probas = KNN_classifer.predict_proba(data_sets)
	build_roc_curve(labels, knn_probas)


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
	print label

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

def print_data(data_sets, labels):
	pd.set_option('display.max_columns', None)
	data_sets["Y"] = labels
	print data_sets.tail(5)
