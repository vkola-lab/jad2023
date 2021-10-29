"""
csv_read.py
read csv files
"""
import csv

def yield_csv_data(csv_in):
	"""
	parameters:
		csv_in(str): path to a csv file
	"""
	with open(csv_in, newline='', encoding='utf-8') as infile:
		for row in csv.DictReader(infile, delimiter=','):
			yield row
