"""
load_all_data.py

load all data into one matrix and then index into that, instead
of always loading data every time we access data;
"""
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
from read_txt import select_task

def combine_arrays(osm, mfcc):
	"""
	combine the arrays;
	"""
	osm_len = len(osm)
	mfcc_len = len(mfcc)
	to_add_len = abs(osm_len - mfcc_len)
	if len(osm) > len(mfcc):
		rows_to_add = np.zeros((to_add_len, mfcc.shape[1]))
		mfcc = np.vstack((mfcc, rows_to_add))
	else:
		rows_to_add = np.zeros((to_add_len, osm.shape[1]))
		osm = np.vstack((osm, rows_to_add))
	return np.append(osm, mfcc, 1)

def load_npy(filepath):
	"""
	load numpy arr
	"""
	np_arr = np.load(filepath)
	if len(np_arr.shape) == 3:
		return np_arr[0]
	return np_arr

def cutoff_array(final_arr, cutoff, win_len=10):
	"""
	final_arr has some shape;
	each item in final_arr represents 10 milliseconds;
	we only want the first cutoff (minutes) of final_arr;
	"""
	windows_per_minute = 60 * 1000 / win_len
	## 60 seconds per min * 1000 milliseconds per second / 10 milliseconds per window
	## equals windows per minute
	up_to = int(windows_per_minute * cutoff)
	assert up_to <= len(final_arr), final_arr.shape
	return final_arr[:up_to]

def shap_load_osm_mfcc(tst_fps_data, cutoff):
	"""
	loading all data - combining osm + mfcc;
	loading from tst_fps_data instead (shap analysis)
	"""
	start = datetime.now()
	print(f'starting to load all data {start};')
	fp_dict = {}
	full_array = None
	idx = 0
	for id_date, data in tqdm(tst_fps_data.items()):
		if data['duration'] < cutoff:
			continue
		fp_tuple = data['osm_fp'], data['mfcc_fp']
		osm_fp, mfcc_fp = fp_tuple
		osm = load_npy(osm_fp)
		mfcc = load_npy(mfcc_fp)
		print(f'osm_shape: {osm.shape}')
		print(f'mfcc_shape: {mfcc.shape}')
		final_arr = combine_arrays(osm, mfcc)
		final_arr = cutoff_array(final_arr, cutoff)
		final_arr = np.expand_dims(final_arr, axis=0)
		## shape = (1, up_to, 18)
		if full_array is None:
			full_array = final_arr
		else:
			full_array = np.concatenate((full_array, final_arr))
		print(f'\tfinal_arr_shape: {final_arr.shape}')
		print(f'\tfull_array_shape: {full_array.shape}')
		print(f'\t{idx}')
		assert fp_tuple not in fp_dict, fp_tuple
		fp_dict[fp_tuple] = {'idx': idx,
			'duration': data['duration'],
			'pt_has_tscript': data['pt_has_tscript'],
			'id_date': id_date}
		idx += 1
	end = datetime.now()
	print(f'loaded {len(fp_dict)} items in {end - start};')
	return fp_dict, full_array

def load_all_osm_and_mfcc(csv_in):
	"""
	loading all data - combining osm + mfcc;
	"""
	final = {}
	for _, row in pd.read_csv(csv_in, dtype=object).iterrows():
		osm_fp = row['osm_npy']
		mfcc_fp = row['mfcc_npy']
		id_date = row['key']
		final[id_date] = (osm_fp, mfcc_fp)
	start = datetime.now()
	print(f'starting to load all data {start};')
	fp_dict = {}
	for _, fp_tuple in tqdm(final.items()):
		osm_fp, mfcc_fp = fp_tuple
		osm = load_npy(osm_fp)
		mfcc = load_npy(mfcc_fp)
		final_arr = combine_arrays(osm, mfcc)
		assert fp_tuple not in fp_dict, fp_tuple
		fp_dict[fp_tuple] = final_arr
	end = datetime.now()
	print(f'loaded {len(fp_dict)} items in {end - start};')
	return fp_dict

def preload_mfcc_load_osm(csv_in):
	"""
		preloading mfcc, live loading OSM
		not enough memory to hold mfcc + OSM
	"""
	final = {}
	for _, row in pd.read_csv(csv_in, dtype=object).iterrows():
		osm_fp = row['osm_npy']
		mfcc_fp = row['mfcc_npy']
		id_date = row['key']
		final[id_date] = (osm_fp, mfcc_fp)
	start = datetime.now()
	print(f'starting to load all data {start};')
	fp_dict = {}
	for _, fp_tuple in tqdm(final.items()):
		osm_fp, mfcc_fp = fp_tuple
		mfcc = load_npy(mfcc_fp)
		assert fp_tuple not in fp_dict, fp_tuple
		fp_dict[fp_tuple] = mfcc
	end = datetime.now()
	print(f'loaded {len(fp_dict)} items in {end - start};')
	return fp_dict

def get_mfcc_osm(fp_tuple, **kwargs):
	"""
	preload mfcc, load osm;
	"""
	all_npy = kwargs['all_npy']
	osm_fp, _ = fp_tuple
	osm_arr = np.load(osm_fp)
	mfcc_arr = all_npy[fp_tuple]
	final_arr = combine_arrays(osm_arr, mfcc_arr)
	return final_arr

def load_all_data(csv_in, audio_idx):
	"""
	loading all data;
	"""
	final = {}
	for _, row in pd.read_csv(csv_in, dtype=object).iterrows():
		audio_fp = row[audio_idx]
		assert audio_fp not in final, audio_fp
		final[audio_fp] = None
	start = datetime.now()
	print(f'starting to load all data {start};')
	for audio_fp, _ in tqdm(final.items()):
		final[audio_fp] = np.load(audio_fp)
		np_arr = final[audio_fp]
		if len(np_arr.shape) == 3:
			final[audio_fp] = final[audio_fp][0]
		print(final[audio_fp].shape)
	end = datetime.now()
	print(f'loaded {len(final)} items in {end - start};')
	return final

def test():
	"""
	testing;
	"""
	csv_info, _ = select_task(1, 'task_csvs.txt')
	return load_all_data(csv_info, 'osm_npy')

if __name__ == '__main__':
	test()
