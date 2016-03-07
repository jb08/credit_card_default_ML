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