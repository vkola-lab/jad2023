"""
compare_vectors.py
"""
import os
import re
import csv
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error, r2_score
from csv_read import yield_csv_data
from get_vols import map_idx_to_region

def write_seed_results(seed_dir, csv_in_list, region):
	"""
	write results to txt;
	"""
	fname = f'txt/{seed_dir.replace("/", "_")}.txt'
	print(seed_dir)
	csv_to_results = {}
	with open(fname, 'w') as outfile:
		all_voice = np.array([])
		all_mri = np.array([])
		all_mse = []
		data_to_write = f'{seed_dir}\n\n'
		for csv_in in csv_in_list:
			voice_arr = np.array([])
			mri_arr = np.array([])
			for row in yield_csv_data(csv_in):
				voice_fsl = np.load(row['voice_fsl_vector_fp'])
				# mri_vector = [(int(i), row[r]) for i, r in idx_to_region.items()]
				# mri_vector.sort()
				mri_vector = np.array([row[region]], dtype='float32')
				voice_arr = np.append(voice_arr, voice_fsl)
				mri_arr = np.append(mri_arr, mri_vector)
			mse = mean_squared_error(mri_arr, voice_arr)
			r_squared = r2_score(mri_arr, voice_arr)
			csv_to_results[csv_in] = {'mse': mse, 'r_squared': r_squared, 'csv_in': csv_in}
			data_to_write += f'\t{os.path.basename(csv_in)}: {mse}\n\n'
			all_mse.append(mse)
			all_voice = np.append(all_voice, voice_arr)
			all_mri = np.append(all_mri, mri_arr)

		outfile.write(get_stats(all_mse, 'mse'))
		outfile.write(get_stats(all_voice, 'voice'))
		outfile.write(get_stats(all_mri, 'mri'))
		outfile.write(data_to_write)
	headers = ['csv_in', 'mse', 'r_squared']
	csv_out_fn = fname.replace('.txt', '.csv')
	with open(csv_out_fn, 'w') as csv_out:
		writer = csv.DictWriter(csv_out, fieldnames=headers)
		writer.writeheader()
		for _, row in csv_to_results.items():
			writer.writerow(row)

def get_stats(data_list, ext):
	"""
	writing min/mean/max
	"""
	min_data = min(data_list)
	mean_data = np.mean(data_list)
	std = np.std(data_list)
	max_data = max(data_list)
	dat = f'\tmin_{ext}: {min_data}\n\n'
	dat += f'\tmean_{ext}: {mean_data}({std})\n\n'
	dat += f'\tmax_{ext}: {max_data}\n\n'
	return f'{ext}\n{dat}--\n'

def eval_all():
	"""
	evaluating all;
	"""
	atlas_xml = 'xml/Hammers_mith_atlases_n30r95_label_indices_SPM12_20170315.xml'
	idx_to_region = map_idx_to_region(atlas_xml)
	ptrn = re.compile(r'fsl_mri_regions\[(\d+)_\]_normalize_fsl_MSELoss')
	parent = 'results/'
	dir_list = [f'{parent}{d}' for d in os.listdir(parent) if ptrn.search(d) is not None]
	result_dirs = {}
	for directory in dir_list:
		region_idx = ptrn.search(directory).groups()[0]
		region = idx_to_region[int(region_idx)]
		if 'normalize_fsl' in directory:
			region += '_brain_frac'
		for epoch_dir in os.listdir(directory):
			epoch_dir = os.path.join(directory, epoch_dir)
			for seed_dir in os.listdir(epoch_dir):
				seed_dir = os.path.join(epoch_dir, seed_dir)
				csv_in_list = [os.path.join(seed_dir, f) for f in os.listdir(seed_dir) if f.endswith('csv')]
				if len(csv_in_list) == 20:
					result_dirs[seed_dir] = (region, csv_in_list)
	for seed_dir, pair in result_dirs.items():
		region, csv_in_list = pair
		write_seed_results(seed_dir, csv_in_list, region)

def main():
	"""
	comparing;
	"""
	atlas_xml = 'xml/Hammers_mith_atlases_n30r95_label_indices_SPM12_20170315.xml'
	idx_to_region = map_idx_to_region(atlas_xml)
	# csv_in_dir = "results/fsl_mri/16_epochs/20982/"
	csv_in_dir = "results/fsl_mri/32_epochs/2878/"
	csv_in_list = [f for f in os.listdir(csv_in_dir) if f.endswith('csv')]
	for csv_fn in csv_in_list:
		print(csv_fn)
		csv_in = os.path.join(csv_in_dir, csv_fn)
		cos_list = []
		for row in yield_csv_data(csv_in):
			voice_fsl = np.load(row['voice_fsl_vector_fp'])
			mri_vector = [(int(i), row[r]) for i, r in idx_to_region.items()]
			mri_vector.sort()
			mri_vector = np.array([r for _, r in mri_vector], dtype='float32')
			# print(voice_fsl.shape)
			# print(mri_vector.shape)
			# for idx, voice_reg in enumerate(voice_fsl):
			# 	print(idx)
			# 	print(voice_reg)
			# 	print(mri_vector[idx])
			# 	input()
			cos = distance.cosine(voice_fsl, mri_vector)
			cos_list.append(cos)
		print(f'min_cos: {min(cos_list)}')
		print(f'mean_cos: {np.mean(cos_list)}')
		print(f'max_cos: {max(cos_list)}')
		print()

if __name__ == '__main__':
	eval_all()
