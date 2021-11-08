"""
mri_func.py
mri-related functions;
"""
import nibabel as nib
import torch
import torch.nn as nn
from csv_read import yield_csv_data
from read_txt import get_pt_file
from net_ie import ImageEncoder
from net_id import ImageDecoder

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
		mni_brain_to_vector[mni_brain] = image_encoder(torch_array).detach().numpy()
	return mni_brain_to_vector

def get_target_vectors(img_pt_idx, img_pt_txt, csv_info):
	"""
	get mri target vectors
	"""
	image_autoencoder = get_autoencoder(img_pt_idx, img_pt_txt)
	loaded_image_encoder = image_autoencoder[0]
	mni_fp_to_vector = gen_mri_latent_vectors(csv_info, loaded_image_encoder)
	return mni_fp_to_vector

def get_autoencoder(img_pt_idx, img_pt_txt):
	"""
	load state dict and get autoencoder;
	"""
	img_pt = get_pt_file(img_pt_idx, img_pt_txt)
	image_encoder = ImageEncoder()
	image_decoder = ImageDecoder()
	image_autoencoder = nn.Sequential(image_encoder, image_decoder)
	img_dict = torch.load(img_pt)['state_dict']
	image_autoencoder.load_state_dict(img_dict)
	return image_autoencoder
