"""
misc.py
miscellaneous helper functions
"""
import os
import random
from datetime import datetime

def get_dir_rsl(ext, num_epochs, seed):
	"""
	small helper;
	"""
	return f'results/{ext}/{num_epochs}_epochs/{seed}'

def gen_seed_dirs(num_seeds, ext, n_epoch):
	"""
	generate seed directories;
	"""
	seed_to_dir = {}
	for _ in range(num_seeds):
		seed = random.randint(0, 100000)
		dir_rsl = get_dir_rsl(ext, n_epoch, seed)
		while os.path.isdir(dir_rsl):
			seed = random.randint(0, 100000)
			dir_rsl = get_dir_rsl(ext, n_epoch, seed)
		os.makedirs(dir_rsl)
		print(dir_rsl)
		seed_to_dir[seed] = dir_rsl
	return seed_to_dir

def get_time():
	"""
	get current time and format it;
	"""
	return str(datetime.now()).replace(' ', '_')
