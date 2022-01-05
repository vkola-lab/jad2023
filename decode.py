"""
decode.py
decode voice-latent vectors into recon'd brains
"""
import numpy as np
import torch
from mri_func import get_autoencoder

def get_image_decoder(img_pt_txt, img_pt_idx):
	"""
	get image decoder;
	"""
	image_autoencoder = get_autoencoder(img_pt_idx, img_pt_txt)
	return image_autoencoder[1]

def decode_npy(image_decoder, npy_fp):
	"""
	decode numpy file;
	"""
	torch_arr = torch.from_numpy(np.load(npy_fp))
	return image_decoder(torch_arr)

def decode(npy_fp):
	"""
	decode a given numpy filepath;
	"""
	return decode_npy(get_image_decoder('img_pt.txt', 0), npy_fp)
