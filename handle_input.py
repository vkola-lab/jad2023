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
	parser.add_argument('-aext', '--arg_ext',
		help='extra string to add to the extension;')
	parser.add_argument('-ai', '--audio_idx',
		help='input to select the type of audio data (osm_npy=0, mfcc=1);')
	parser.add_argument('-ipt', '--img_pt_txt',
		help='path to the img autoencoder pt txt file')
	parser.add_argument('-img_pt', '--img_pt_idx',
		help='input to select a img autoencoder pretrained (pt) file;')
	parser.add_argument('-enc', '--encode_idx',
		help='0=CNN image autoencoder, 1=GAN image autoencoder')
	parser.add_argument("-d", "--device", help='gpu device index;')
	parser.add_argument('-ne', '--n_epoch', help='indicates number of epochs;')
	parser.add_argument('-ns', '--num_seeds', help='indicates number of seeds to use;')
	parser.add_argument("-nf", "--num_folds", help='number of cross validation folds;')
	parser.add_argument('-ht', '--holdout_test', action='store_true',
		help='if set, test fold is held static;')
	parser.add_argument('-drs', '--do_rand_seg', action='store_true',
		help='if set, use random segments instead of static consecutive segments')
	parser.add_argument('-nptseg', '--num_pt_segments', help='number of random pt segments')
	parser.add_argument('-ptsegroot', '--pt_segment_root', help='root to save random pt segments')
	parser.add_argument('-segmin', '--seg_min', help='duration of each random pt segment')
	parser.add_argument('-lr', '--learning_rate',
		help='assign the learning rate, default is 1e-4;')
	parser.add_argument('-db', '--debug_stop', action='store_true',
		help='if set, execution is stopped short for debugging;')
	parser.add_argument('-nlw', '--negative_loss_weight',
		help='loss weight for negative label;')
	parser.add_argument('-plw', '--positive_loss_weight',
		help='loss weight for positive label;')
	parser.add_argument('-nsm', '--no_save_model', action='store_true',
		help='if set, models will not be saved;')
	parser.add_argument('-nwft', '--no_write_fold_txt', action='store_true',
		help='if set, fold txt files are not written;')
	parser.add_argument('-ri', '--region_indices', nargs='*',
		help='region indices to use, default is all regions')
	return parser.parse_args(args).__dict__
