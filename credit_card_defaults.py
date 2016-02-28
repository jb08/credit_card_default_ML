import pandas as pd
import sys
from pandas import DataFrame


def csv_reader(file_name):
	
	frame = pd.read_csv(file_name, index_col = 0, header = 0, skiprows = [1])
	return frame

def main():
	path = "default of credit card clients.csv";
	frame = csv_reader(path)
	print frame.head(10)

if __name__ == "__main__":
	main()


