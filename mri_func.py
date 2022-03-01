"""
mri_func.py
mri-related functions;
"""
import nibabel as nib
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import image_adversarial_autoencoder as iaa
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

def get_gan_autoencoder(img_pt_idx, img_pt_txt):
	"""
	load state and get gan autoencoder;
	"""
	img_pt = get_pt_file(img_pt_idx, img_pt_txt)
	autoencoder = iaa.ImageAdversarialAutoencoder_64()
	img_dict = torch.load(img_pt)
	autoencoder.load_state_dict(img_dict)
	return autoencoder

def gan_get_target_vectors(img_pt_idx, img_pt_txt, csv_info):
	"""
	get mri target vectors with GAN network;
	"""
	image_autoencoder = get_gan_autoencoder(img_pt_idx, img_pt_txt)
	encoder = image_autoencoder.encoder
	mni_fp_to_vector = gan_gen_mri_latent_vectors(csv_info, encoder)
	return mni_fp_to_vector

def gan_downsample_mni(fname):
	"""
	downsample mni brain for gan encoding
	"""
	img = np.array(nib.load(fname).dataobj)
	img = img[:, 18:-18, :]
	img = ndimage.zoom(img, 0.352)
	torch_img = torch.from_numpy(img)
	torch_img = torch.unsqueeze(torch_img, 0)
	torch_img = torch.unsqueeze(torch_img, 0)
	return torch_img

def gan_gen_mri_latent_vectors(csv_in, image_encoder):
	"""
	generate the mri latent vectors
	"""
	mni_brain_to_vector = {}
	mni_brains = set()
	for row in yield_csv_data(csv_in):
		mni_brains.add(row['mni_brain'])
	for mni_brain in mni_brains:
		torch_img = gan_downsample_mni(mni_brain)
		mni_brain_to_vector[mni_brain] = image_encoder(torch_img).detach().numpy()
	return mni_brain_to_vector
