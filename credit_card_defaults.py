import pandas as pd
import sys
from pandas import DataFrame
from classifier import build_KNN_classifier

def csv_reader():
	file_name = "default of credit card clients.csv"
	data_sets = pd.read_csv(file_name, index_col = 0, header = 0, skiprows = [1], usecols= range(24))
	labels = pd.read_csv(file_name, index_col = 0, header = 0, skiprows = [1], usecols= [0,24])
	return data_sets, labels

def main():
	data_sets, labels = csv_reader()
	#print data_sets.head(5)
	#print "------"
	#print labels.head(5)

	classifer = build_KNN_classifier(data_sets, labels)
	

if __name__ == "__main__":
	main()