"""
handle_input.py
misc functions for giving input to CNN model running;
"""
import argparse

def get_args(args):
	"""
	set variables based on cmd line args;
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('-tct', '--task_csv_txt',
		help='path to the task csv txt file, defaults to task_csv.txt')
	parser.add_argument('-ti', '--task_id',
		help='input to select a task, which selects a csv, ext, and get_label;')
	return parser.parse_args(args).__dict__
