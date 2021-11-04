"""
Created on Wed Oct 27 11:09:36 2021

@author: cxue2
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import os
import sys
import nibabel as nib
import torch
import torch.nn as nn
from audio_dataset import AudioDataset
from csv_read import yield_csv_data
from handle_input import get_args
from misc import gen_seed_dirs, get_time
from net_ie import ImageEncoder
from net_id import ImageDecoder
from net_ve import VoiceEncoder
from read_txt import select_task, get_pt_file

def pad_mni_brain(mni_brain):
	"""
	padding mni brain to proper dimensions;
	"""
	torch_array = torch.from_numpy(nib.load(mni_brain).get_fdata())
	height, width, depth = torch_array.shape
	torch_array = torch.unsqueeze(torch_array, 0)
	torch_array = torch.unsqueeze(torch_array, 0)
	torch_array = torch_array[:,:,0:(height - 1),0:(width-2),0:(depth-1)]
	return nn.functional.pad(torch_array, (4, 4, 0, 0, 4, 4), "constant", 0)

def gen_mri_latent_vectors(csv_in, image_encoder):
	"""
	generate the mri latent vectors
	"""
	mni_brain_to_vector = {}
	mni_brains = set()
	for row in yield_csv_data(csv_in):
		mni_brains.add(row['mni_brain'])
	for mni_brain in mni_brains:
		torch_array = pad_mni_brain(mni_brain).to(torch.float)
		mni_brain_to_vector[mni_brain] = image_encoder(torch_array)
	return mni_brain_to_vector

def get_target_vectors(img_pt_idx, img_pt_txt, csv_info):
	"""
	get mri target vectors
	"""
	img_pt = get_pt_file(img_pt_idx, img_pt_txt)
	image_encoder = ImageEncoder()
	image_decoder = ImageDecoder()
	image_autoencoder = nn.Sequential(image_encoder, image_decoder)
	img_dict = torch.load(img_pt)['state_dict']
	image_autoencoder.load_state_dict(img_dict)
	loaded_image_encoder = image_autoencoder[0]
	mni_fp_to_vector = gen_mri_latent_vectors(csv_info, loaded_image_encoder)
	# for mni_fp, mni_vector in mni_fp_to_vector.items():
	# 	print(mni_fp)
	# 	print(mni_vector.shape)
	return mni_fp_to_vector

def main():
	"""
	main entrypoint;
	"""
	args = get_args(sys.argv[1:])
	args = {k: v for k, v in args.items() if v is not None}
	task_csv_txt = args.get('task_csv_txt', 'task_csvs.txt')
	task_id = args.get('task_id', 0)
	img_pt_txt = args.get('img_pt_txt', 'img_pt.txt')
	img_pt_idx = args.get('img_pt_idx', 0)
	device = int(args.get('device', 0))
	n_epoch = int(args.get('n_epoch', 1))
	num_seeds = int(args.get('num_seeds', 1))
	num_folds = int(args.get('num_folds', 5))
	csv_info, ext = select_task(task_id, task_csv_txt)
	mni_fp_to_vector = get_target_vectors(img_pt_idx, img_pt_txt, csv_info)

	seed_to_dir = gen_seed_dirs(num_seeds, ext, n_epoch)

	for seed, dir_rsl in seed_to_dir.items():
		time_ext = get_time()
		dir_rsl = f'{dir_rsl}/{time_ext}'
		assert not os.path.isdir(dir_rsl), dir_rsl
		os.makedirs(dir_rsl)
		print(dir_rsl)
		for vld_idx in range(num_folds):
			for tst_idx in range(num_folds):
				if vld_idx == tst_idx:
					continue
					## vld and tst fold can't be the same?
				dset_kw = {'num_folds': num_folds, 'vld_idx': vld_idx, 'tst_idx': tst_idx,
					'seed': seed, 'mni_fp_to_vector': mni_fp_to_vector}
				dset_trn = AudioDataset(csv_info, 'TRN', **dset_kw)
				dset_vld = AudioDataset(csv_info, 'VLD', **dset_kw)
				dset_tst = AudioDataset(csv_info, 'TST', **dset_kw)

if __name__ == '__main__':
	main()
