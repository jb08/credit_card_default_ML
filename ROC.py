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

	#KNN
	KNN_classifier = build_KNN_classifier(data_sets, labels)
	KNN_predicted = predict_test_data(data_sets, KNN_classifier)
	knn_probas = KNN_classifier.predict_proba(data_sets)

	#LR
	LR_classifier = build_LR_classifier(data_sets, labels)
	LR_predicted = predict_test_data(data_sets, LR_classifier)
	LR_probas = LR_classifier.predict_proba(data_sets)

	#DA
	DA_classifier = build_DA_classifier(data_sets, labels)
	DA_predicted = predict_test_data(data_sets, DA_classifier)
	DA_probas = DA_classifier.predict_proba(data_sets)

	#DT
	DT_classifier = build_DT_classifier(data_sets, labels)
	DT_predicted = predict_test_data(data_sets, DT_classifier)
	DT_probas = DT_classifier.predict_proba(data_sets)

	#NB
	NB_classifier = build_NB_classifier(data_sets, labels)
	NB_predicted = predict_test_data(data_sets, NB_classifier)
	NB_probas = NB_classifier.predict_proba(data_sets)

	print_error_rates = False
	if(print_error_rates):
		print_error_rate("KNN", KNN_predicted, labels)
		print_error_rate("LR", LR_predicted, labels)
		print_error_rate("DA", DA_predicted, labels)
		print_error_rate("DT", DT_predicted, labels)
		print_error_rate("NB", NB_predicted, labels)

	#ROC analysis
	build_roc_curve(labels, knn_probas, LR_probas, DA_probas, DT_probas, NB_probas)


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

def build_roc_curve(labels, knn_probas, LR_probas, DA_probas, DT_probas, NB_probas):

	knn_fpr, knn_tpr, knn_thresholds = roc_curve(labels, knn_probas[:, 1])
	knn_roc_auc = auc(knn_fpr, knn_tpr)
	knn_output=('KNN AUC = %0.2f'% knn_roc_auc)
	print knn_output

	LR_fpr, LR_tpr, LR_thresholds = roc_curve(labels, LR_probas[:, 1])
	LR_roc_auc = auc(LR_fpr, LR_tpr)
	LR_output=('LR AUC = %0.2f'% LR_roc_auc)
	print LR_output

	DA_fpr, DA_tpr, DA_thresholds = roc_curve(labels, DA_probas[:, 1])
	DA_roc_auc = auc(DA_fpr, DA_tpr)
	DA_output=('DA AUC = %0.2f'% DA_roc_auc)
	print DA_output

	DT_fpr, DT_tpr, DT_thresholds = roc_curve(labels, DT_probas[:, 1])
	DT_roc_auc = auc(DT_fpr, DT_tpr)
	DT_output=('DT AUC = %0.2f'% DT_roc_auc)
	print DT_output

	NB_fpr, NB_tpr, NB_thresholds = roc_curve(labels, NB_probas[:, 1])
	NB_roc_auc = auc(NB_fpr, NB_tpr)
	NB_output=('NB AUC = %0.2f'% NB_roc_auc)
	print NB_output
	
	plot_on = True
	if(plot_on):
		#setup plot
		plt.plot(DT_fpr, DT_tpr, label='Classification tree')
		plt.plot(knn_fpr, knn_tpr, label='KNN')
		plt.plot(NB_fpr, NB_tpr, label='Naive Bayesian')
		plt.plot(DA_fpr, DA_tpr, label='Discriminant Analysis')
		plt.plot(LR_fpr, LR_tpr, label='LogRegression')
		
		plt.axis([-.1, 1, 0, 1.1])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver Operating Characteristic')
		plt.legend(loc="lower right")
		plt.show()

def calc_confusion_matrix(model, predicted, labels):
	cm = confusion_matrix(labels, predicted, labels = [0,1])

	print model + " confusion_matrix: "
	print cm
	print "---"

def print_data(data_sets, labels):
	pd.set_option('display.max_columns', None)
	data_sets["Y"] = labels
	print data_sets.tail(5)

def print_error_rate(model, predicted, labels):
	error_rate = error_measure(predicted, labels)
	print model + " error rate: ", error_rate