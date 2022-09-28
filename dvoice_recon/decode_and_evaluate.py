"""
decode_and_evaluate.py
decode and evaluate MRI latent vectors produced by the VoiceEncoder(),
based on the MRI image CNN-based autoencoder
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from csv_read import yield_csv_data
from mri_func import pad_mni_brain, get_autoencoder

def decode_npy(image_decoder, npy_fp):
	"""
	decode numpy file;
	"""
	torch_arr = torch.from_numpy(np.load(npy_fp))
	return image_decoder(torch_arr)

def plot_slice(arr, idx, ext):
	"""
	squeeze and plot;
	"""
	one_slice = arr[idx,:,:]
	print(one_slice.shape)
	plt.imshow(one_slice, cmap='gray')
	print(idx)
	plt.savefig(f'{ext}/{idx}.png')

def read_from_csv(csv_in, cog):
	"""
	get first set of paths from first instance with cog=cog;
	"""
	cog = float(cog)
	for row in yield_csv_data(csv_in):
		if float(row['cog']) == cog:
			return row['mni_fp'], row['voice_latent']
	return None, None

def main():
	"""
	main entrypoint;
	"""
	csv_in = "cnn_autoencoder_data.csv"
	norm_mni, norm_npy_fp = read_from_csv(csv_in, 0)
	normal = (norm_mni, norm_npy_fp, 'norm')
	## normal brain;

	mci_mni, mci_npy_fp = read_from_csv(csv_in, 0.5)
	mci = (mci_mni, mci_npy_fp, 'mci')
	## mci brain

	ad_mni, ad_npy_fp = read_from_csv(csv_in, 1)
	ad_pair = (ad_mni, ad_npy_fp, 'ad')
	## ad brain

	orig_mni, npy_fp, ext = normal
	# orig_mni, npy_fp, ext = mci
	# orig_mni, npy_fp, ext = ad_pair

	tmp_dir = f'tmp_{ext}'
	tmp_decoded_dir = f'tmp_decoded_{ext}'
	if not os.path.isdir(tmp_dir):
		os.makedirs(tmp_dir)
	if not os.path.isdir(tmp_decoded_dir):
		os.makedirs(tmp_decoded_dir)

	img_pt_idx = 0
	img_pt_txt = "img_pt.txt"
	image_autoencoder = get_autoencoder(img_pt_idx, img_pt_txt)
	image_decoder = image_autoencoder[1]
	mni_arr = pad_mni_brain(orig_mni)

	decoded_arr = decode_npy(image_decoder, npy_fp)
	print(decoded_arr.shape)
	print(mni_arr.shape)
	print('orig mni')
	mni_arr = torch.squeeze(mni_arr).numpy()
	num_slices = mni_arr.shape[0]
	for idx in range(num_slices):
		if os.path.isfile(f'{tmp_dir}/{idx}.png'):
			continue
		plot_slice(mni_arr, idx, tmp_dir)
	print()

	print('decoded_arr')
	decoded_arr = torch.squeeze(decoded_arr).detach().numpy()

	num_slices = decoded_arr.shape[0]
	for idx in range(num_slices):
		plot_slice(decoded_arr, idx, tmp_decoded_dir)

if __name__ == '__main__':
	main()
