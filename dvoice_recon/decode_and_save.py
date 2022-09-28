"""
decode_and_save.py
decode voice-latent vectors into reconstructed brains and save as a numpy array;
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from csv_read import yield_csv_data
from decode import decode

def decode_and_save(csv_in):
	"""
	csv_in: cols=patient_id, audio_fn, mni_brain, start, end, voice_mri_vector_fp
	"""
	list_of_rows = []
	for row in yield_csv_data(csv_in):
		npy_fp = row['voice_mri_vector_fp']
		parent_dir = os.path.dirname(npy_fp)

		decoded_parent = os.path.join(parent_dir, 'decoded_voice_brains')
		assert '_mfcc' in os.path.basename(npy_fp), npy_fp
		decoded_fn = os.path.basename(npy_fp).replace('_mfcc_', '_recon_brain_')
		decoded_fp = os.path.join(decoded_parent, decoded_fn)

		decoded_array = decode(npy_fp)
		decoded_array = torch.squeeze(decoded_array).detach().numpy()
		if not os.path.isdir(decoded_parent):
			os.makedirs(decoded_parent)
		if not os.path.isfile(decoded_fp):
			np.save(decoded_fp, decoded_array)
			print(f'saved: {decoded_fp};')
		row['decoded_voice_recon_brain'] = decoded_fp
		list_of_rows.append(row)
	return list_of_rows

def main(csv_in):
	"""
	main entrypoint;
	"""
	csv_parent = os.path.dirname(csv_in)
	csv_out_parent = os.path.join(csv_parent, 'decoded_csvs')
	csv_out_fn = f'decoded_{os.path.basename(csv_in)}'

	csv_out = os.path.join(csv_out_parent, csv_out_fn)

	list_of_rows = decode_and_save(csv_in)

	if not os.path.isdir(csv_out_parent):
		os.makedirs(csv_out_parent)

	pd.DataFrame(list_of_rows).to_csv(csv_out, index=False)
	print(csv_out)

if __name__ == '__main__':
	main(sys.argv[1])
