import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
from classifier import *

def csv_reader():
	file_name = "default of credit card clients.csv"
	data_sets = pd.read_csv(file_name, index_col = 0, header = 0, skiprows = [1], usecols= range(24))
	labels = pd.read_csv(file_name, index_col = 0, header = 0, skiprows = [1], usecols= [0,24])
	return data_sets, labels

def main():
	data_sets, labels = csv_reader()
	labels = np.ravel(labels)

	KNN_classifer = build_KNN_classifier(data_sets, labels)
	KNN_predicted = predict_test_data(data_sets, KNN_classifer)
	KNN_error_rate = error_measure(KNN_predicted, labels)
	print "KNN_error_rate: ",  KNN_error_rate

	LR_classifer = build_LR_classifier(data_sets, labels)
	LR_predicted = predict_test_data(data_sets, LR_classifer)
	LR_error_rate = error_measure(LR_predicted, labels)
	print "LR_error_rate: ",  LR_error_rate

	DA_classifer = build_DA_classifier(data_sets, labels)
	DA_predicted = predict_test_data(data_sets, DA_classifer)
	DA_error_rate = error_measure(DA_predicted, labels)
	print "DA_error_rate: ",  DA_error_rate

	NB_classifer = build_NB_classifier(data_sets, labels)
	NB_predicted = predict_test_data(data_sets, NB_classifer)
	NB_error_rate = error_measure(NB_predicted, labels)
	print "NB_error_rate: ",  NB_error_rate

	NN_classifer = build_NN_classifier(data_sets, labels)
	NN_predicted = predict_test_data(data_sets, NN_classifer)
	NN_error_rate = error_measure(NN_predicted, labels)
	print "NN_error_rate: ",  NN_error_rate

if __name__ == "__main__":
	main()