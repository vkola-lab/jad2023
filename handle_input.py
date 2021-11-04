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
	parser.add_argument('-ipt', '--img_pt_txt',
		help='path to the img autoencoder pt txt file')
	parser.add_argument('-img_pt', '--img_pt_idx',
		help='input to select a img autoencoder pretrained (pt) file;')
	parser.add_argument("-d", "--device", help='gpu device index;')
	parser.add_argument('-ne', '--n_epoch', help='indicates number of epochs;')
	parser.add_argument('-ns', '--num_seeds', help='indicates number of seeds to use;')
	parser.add_argument("-nf", "--num_folds", help='number of cross validation folds;')
	return parser.parse_args(args).__dict__