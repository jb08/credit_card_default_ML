import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
from classifier import build_KNN_classifier, error_measure, predict_test_data

def csv_reader():
	file_name = "default of credit card clients.csv"
	data_sets = pd.read_csv(file_name, index_col = 0, header = 0, skiprows = [1], usecols= range(24))
	labels = pd.read_csv(file_name, index_col = 0, header = 0, skiprows = [1], usecols= [0,24])
	return data_sets, labels

def main():
	data_sets, labels = csv_reader()
	labels = np.ravel(labels)
	#print labels
	# print data_sets
	classifer = build_KNN_classifier(data_sets, labels)
	predicted = predict_test_data(data_sets, classifer)
	error_rate = error_measure(predicted, labels)
	print error_rate

if __name__ == "__main__":
	main()