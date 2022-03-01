"""
gan_decode_and_evaluate.py
decode and evaluate MRI latent vectors produced by the VoiceEncoder(),
but using the GAN autoencoder
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from csv_read import yield_csv_data
from mri_func import gan_downsample_mni, get_gan_autoencoder

def decode_npy(image_decoder, npy_fp):
	"""
	decode numpy file;
	"""
	torch_arr = torch.from_numpy(np.load(npy_fp))
	torch_arr = torch.unsqueeze(torch_arr, 0)
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
			return row['mni_fp'], row['voice_latent'], row['id_date']
	return None, None

def main():
	"""
	main entrypoint;
	"""
	csv_in = "gan_autoencoder_data.csv"
	norm_mni, norm_npy_fp, id_date = read_from_csv(csv_in, 0)
	normal = (norm_mni, norm_npy_fp, 'norm')
	## normal brain;

	orig_mni, npy_fp, ext = normal

	tmp_dir = f'gan/orig_mni_{ext}/{id_date}'
	tmp_decoded_dir = f'gan/decoded_voice_{ext}/{id_date}'
	if not os.path.isdir(tmp_dir):
		os.makedirs(tmp_dir)
	if not os.path.isdir(tmp_decoded_dir):
		os.makedirs(tmp_decoded_dir)

	img_pt_idx = 0
	img_pt_txt = "gan_pt.txt"
	image_autoencoder = get_gan_autoencoder(img_pt_idx, img_pt_txt)
	image_decoder = image_autoencoder.decoder
	mni_arr = gan_downsample_mni(orig_mni)

	decoded_arr = decode_npy(image_decoder, npy_fp)
	print(decoded_arr.shape)
	print(mni_arr.shape)

	print('orig mni')
	mni_arr = torch.squeeze(mni_arr).numpy()
	# np.save(f'{tmp_dir}/{id_date}_orig_mni.npy', mni_arr)
	# print('saved mni arr')
	num_slices = mni_arr.shape[0]
	for idx in range(num_slices):
		if os.path.isfile(f'{tmp_dir}/{idx}.png'):
			continue
		plot_slice(mni_arr, idx, tmp_dir)
	print()

	print('decoded_arr')
	decoded_arr = torch.squeeze(decoded_arr).detach().numpy()
	# np.save(f'{tmp_decoded_dir}/{id_date}_decoded_voice.npy', decoded_arr)
	# print('saved decoded arr')
	num_slices = decoded_arr.shape[0]
	for idx in range(num_slices):
		plot_slice(decoded_arr, idx, tmp_decoded_dir)

if __name__ == '__main__':
	main()
