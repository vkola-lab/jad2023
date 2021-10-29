"""
Created on Wed Oct 27 11:09:36 2021

@author: cxue2
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import sys
import nibabel as nib
import torch
import torch.nn as nn
from csv_read import yield_csv_data
from handle_input import get_args
from select_task import select_task
from net_ie import ImageEncoder
from net_id import ImageDecoder
from net_ve import VoiceEncoder

def gen_mri_latent_vectors(csv_in, image_encoder):
	"""
	generate the mri latent vectors
	"""
	mni_brain_to_vector = {}
	mni_brains = set()
	for row in yield_csv_data(csv_in):
		mni_brains.add(row['mni_brain'])
	for mni_brain in mni_brains:
		torch_array = torch.from_numpy(nib.load(mni_brain).get_fdata())
		mni_brain_to_vector[mni_brain] = image_encoder(torch_array)
	return mni_brain_to_vector

# generate feature/latent vectors using image encoder
# don't forget to DETACH the outputs using Tensor.detach()

# new voice encoder
# net_ve = VoiceEncoder()

# train voice encoder

# save
# torch.save({k: v.cpu() for k, v in net_ve.state_dict().items()}, './save/exp_1/net_ve.pt')

def main():
	"""
	main entrypoint;
	"""
	args = get_args(sys.argv[1:])
	args = {k: v for k, v in args.items() if v is not None}
	task_csv_txt = args.get('task_csv_txt', 'task_csvs.txt')
	task_id = args.get('task_id', 0)
	csv_info, ext = select_task(task_id, task_csv_txt)

	image_encoder = ImageEncoder()
	image_decoder = ImageDecoder()
	image_autoencoder = nn.Sequential(image_encoder, image_decoder)
	image_autoencoder.load_state_dict(torch.load('./save/exp_1/net_ie.pt'))

	loaded_image_encoder = image_autoencoder[0]
	mni_brain_to_vector = gen_mri_latent_vectors(csv_info, loaded_image_encoder)
	for mni_brain, mni_vector in mni_brain_to_vector.items():
		print(mni_brain)
		print(mni_vector.shape)

if __name__ == '__main__':
	main()
