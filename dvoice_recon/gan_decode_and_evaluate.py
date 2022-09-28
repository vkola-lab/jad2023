"""
gan_decode_and_evaluate.py
decode and evaluate MRI latent vectors produced by the VoiceEncoder(),
but using the GAN autoencoder
"""
import os
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
from csv_read import yield_csv_data
from read_txt import select_task, get_tst_csv
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

def write_csv(final, headers, csv_out):
	"""
	write csv
	"""
	with open(csv_out, 'w', newline='') as outfile:
		writer = csv.DictWriter(outfile, fieldnames=headers)
		writer.writeheader()
		for data in final:
			writer.writerow(data)
	print(f'wrote: {csv_out}')

def get_voice_vectors(tst_csv):
	"""
	tst_csv: patient_id, audio_fn, mni_brain, start, end, voice_mri_vector_fp

	return dictionary keyed by audio_fn: [list_of_rows]
	"""
	final = defaultdict(list)
	for row in yield_csv_data(tst_csv):
		final[row['audio_fn']].append(row)
	return final

def get_phenotypic(task_csv):
	"""
	task_csv: idtype, id, date, id_date, mri_date, mri_diff,
		is_norm, is_mci, is_demented, is_ad, has_transcript_and_nearby_mri, pt_npy,
		tscript, mni_brain

	return dictionary keyed by pt_npy: [{idtype, id, date, id_date, is_norm, is_mci,
		is_demented, is_ad}]
	"""
	final = defaultdict(list)
	for row in yield_csv_data(task_csv):
		final[row['pt_npy']] = {k: row[k] for k in ['idtype', 'id', 'date', 'id_date',
			'is_norm', 'is_mci', 'is_demented', 'is_ad']}
	return final

def recon_voice(voice_vector, image_decoder, recon_voice_fp):
	"""
	reconstruct MRI from voice vector;

	voice_vector: latent voice vector
	"""
	decoded_voice = decode_npy(image_decoder, voice_vector)
	decoded_voice = torch.squeeze(decoded_voice).detach().numpy()
	np.save(recon_voice_fp, decoded_voice)
	print(f'saved {recon_voice_fp};')
	return decoded_voice.shape

def main():
	"""
	2022-03-22
	main entrypoint;
	"""
	tst_csv = get_tst_csv(0, 'tst_csvs.txt')
	task_csv, _ = select_task(0, 'task_csvs.txt')

	whole_audio_to_voice_vectors = get_voice_vectors(tst_csv)
	whole_audio_to_phenotypic = get_phenotypic(task_csv)

	image_autoencoder = get_gan_autoencoder(0, 'gan_pt.txt')
	image_decoder = image_autoencoder.decoder

	parent_dir = os.path.join(os.path.dirname(tst_csv), 'recon')
	out_fn = f'recon_{os.path.basename(tst_csv)}'
	csv_out = os.path.join(parent_dir, out_fn)
	if not os.path.isdir(parent_dir):
		os.makedirs(parent_dir)

	recon_parent_dir = os.path.join(parent_dir, os.path.splitext(out_fn)[0])
	if not os.path.isdir(recon_parent_dir):
		os.makedirs(recon_parent_dir)

	final = []
	for whole_audio, list_of_voice_vector_data in whole_audio_to_voice_vectors.items():
		pheno_data = whole_audio_to_phenotypic[whole_audio]
		for voice_vector_data in list_of_voice_vector_data:
			new_data = dict(pheno_data)
			new_data.update({k: voice_vector_data[k] for k in ['mni_brain', 'start', 'end',
				'voice_mri_vector_fp']})
			voice_vector = voice_vector_data['voice_mri_vector_fp']
			recon_fn = os.path.basename(voice_vector).replace('voice_mri_mfcc', 'voice_mri_recon')
			recon_voice_fp = os.path.join(recon_parent_dir, recon_fn)
			recon_shape = recon_voice(voice_vector, image_decoder, recon_voice_fp)
			new_data.update({'recon_voice_fp': recon_voice_fp, 'recon_shape': recon_shape})
			final.append(new_data)

	headers = ['idtype', 'id', 'date', 'id_date', 'mni_brain', 'recon_voice_fp', 'recon_shape',
		'voice_mri_vector_fp', 'is_norm', 'is_mci', 'is_demented', 'is_ad', 'start', 'end']
	write_csv(final, headers, csv_out)

def _old_main():
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
