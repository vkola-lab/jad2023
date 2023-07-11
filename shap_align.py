"""
shap_align.py
align a shap array
"""
import os
import csv
import subprocess
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from csv_read import yield_csv_data
from shap_analyze import get_tst_fps_data, get_id_date_to_dur, json_load
from misc import get_time

def get_shap_fp(id_date, results_dir, vld_idx, tst_idx, cutoff):
	"""
	get shap filepath
	"""
	shap_parent = os.path.join(results_dir, f'shap/{cutoff}/2023_02_02/' +\
		f'{vld_idx}_{tst_idx}/')
	shap_fn = f'{id_date}_[{cutoff}]_vld_{vld_idx}_tst_{tst_idx}_pos.npy'
	return os.path.join(shap_parent, shap_fn)

def as_id_date(data_json):
	"""
	rekey as id_date
	"""
	final = {}
	for _, list_of_data in json_load(data_json).items():
		for data in list_of_data:
			assert data['id_date'] not in final, data['id_date']
			final[data['id_date']] = data
	return final

def list_to_csv(csv_out, headers, final, extrasaction='raise'):
	"""
	write list to csv
	"""
	if not os.path.isdir(os.path.dirname(csv_out)):
		os.makedirs(os.path.dirname(csv_out))
	with open(csv_out, 'w', newline='') as outfile:
		writer = csv.DictWriter(outfile, fieldnames=headers, extrasaction=extrasaction)
		writer.writeheader()
		for data in final:
			writer.writerow(data)

def config():
	"""
	defining different files and configurations
	"""
	results_dir = "results/HC_vs_DE_tscript_osm_and_mfcc_npy_1.0_1.0_lr_nworkers0_orig_tcn/"+\
		"32_epochs/"+\
		"21639/"
	vld_idx, tst_idx, cutoff = "0", "1", 5
	json_in = "json_in/"+\
		"remap_(760)_[1511]_20221021_14_13_54_0011_osm_dr_longest_mfccs_osm_npy_tscript.json"
	tscript = "json_in/"+\
		"iterative_reduce_(656)_[1264]_20210519_11_10_20_0386_attach_transcripts.json"
	id_date_tscript = as_id_date(tscript)
	id_date_to_dur = get_id_date_to_dur(json_in)
	return results_dir, vld_idx, tst_idx, cutoff, id_date_tscript, id_date_to_dur

def set_tst_fps_data(vld_idx, tst_idx, results_dir, id_date_to_dur):
	"""
	getting tst fps data
	"""
	csv_fn = f"tst_audio_21639_vld_{vld_idx}_tst_{tst_idx}.csv"
	csv_in = os.path.join(results_dir, csv_fn)

	kwargs = {'add_keys': ['label', 'score']}
	return get_tst_fps_data(csv_in, id_date_to_dur, limit=None, **kwargs), csv_fn

def map_shap_to_sec(shap_arr):
	"""
	for a given shap array, map its index to its corresponding starting
	and ending second;
	"""
	idx_to_sec = {}
	## convert from (1, 1, 18, 30000) to (18, 30000)
	shap_arr = np.swapaxes(np.squeeze(np.load(shap_arr)), 0, 1)
	for idx, _ in enumerate(shap_arr):
		start_sec = np.floor(idx / 100)
		idx_to_sec[idx] = start_sec
	return idx_to_sec, shap_arr

def timestamp_to_sec(data):
	"""
	hh mm ss to seconds
	"""
	return int(data['hour']) * 60 * 60 + int(data['minute']) * 60 + int(data['second'])

def read_tscript_csv(tscript_csv):
	"""
	read in data from tscript csv
	"""
	final = {}
	for row in yield_csv_data(tscript_csv):
		start_sec = timestamp_to_sec(row)
		end_sec = start_sec + int(float(row['duration']))
		final[(start_sec, end_sec)] = [row['speaker'], row['segment_name']]
	final_keys = list(final.keys())
	final_keys.sort()
	final = {k: final[k] for k in final_keys}
	return final

def save_shap_map(shap_fp, tscript_csv):
	"""
	save a shap numpy array with the same size
	as the shap_fp array, but with contents
	that indicate the speaker and segment_name
	"""
	idx_to_sec, shap_arr = map_shap_to_sec(shap_fp)
	tscript_seconds_to_data = read_tscript_csv(tscript_csv)

	shap_map = np.zeros((len(shap_arr), 2)).astype('object')
	not_found = 0
	for idx, shap_start_sec in idx_to_sec.items():
		found = False
		for pair, tscript_data in tscript_seconds_to_data.items():
			start, end = pair
			if start <= shap_start_sec <= end:
				shap_map[idx] = tscript_data
				found = True
				break
		if not found:
			not_found += 1
	print(f'not_found: {not_found}')
	shap_map_fp = os.path.join(os.path.dirname(shap_fp),
		f'shap_map/shap_map_{os.path.basename(shap_fp)}')
	print(f'saving {shap_map_fp}')
	if not os.path.isdir(os.path.dirname(shap_map_fp)):
		os.makedirs(os.path.dirname(shap_map_fp))
	np.save(shap_map_fp, shap_map)
	return shap_map_fp, shap_map, shap_arr

def count_and_write_csv(tst_fps_data, cutoff, results_dir, vld_idx,
	tst_idx, id_date_tscript, csv_fn):
	"""
	count tscript_lens and write shap csv
	"""
	count = defaultdict(int)
	final = []
	for id_date, data in tst_fps_data.items():
		if float(data['duration']) >= cutoff and int(data['pt_has_tscript']) == 1:
			shap_fp = get_shap_fp(id_date, results_dir, vld_idx, tst_idx, cutoff)
			assert os.path.isfile(shap_fp), shap_fp
			tscript_len = id_date_tscript.get(id_date, {}).get('duration_csv_out_list_len')
			## pt_has_tscript means they have at least one tscript
			## not that the id_date has a tscript
			if tscript_len is not None and int(tscript_len) == 1:
				## take those with 1 tscript for simplicity
				duration_csv = id_date_tscript[id_date]['duration_csv_out_list'][0]
				duration_csv = duration_csv.replace("\\", "/")
				assert os.path.isfile(duration_csv), duration_csv
				data.update({'shap_fp': shap_fp, 'cutoff': cutoff, 'vld_idx': vld_idx,
					'tst_idx': tst_idx, 'id_date': id_date,
					'duration_csv': duration_csv})
				final.append(data)
				shap_map_fp, shap_map, shap_arr = save_shap_map(shap_fp, duration_csv)
				print(shap_map.shape)
				print(shap_arr.shape)
				print()
				data['shap_map_fp'] = shap_map_fp
			count[tscript_len] += 1
	csv_out = os.path.join(results_dir, f'shap_csv/shap_map_{get_time()}_{csv_fn}')
	headers = ['id_date', 'duration', 'pt_has_tscript', 'cutoff',
		'vld_idx', 'tst_idx', 'label', 'score', 'shap_map_fp', 'shap_fp',
		'duration_csv', 'osm_fp', 'mfcc_fp',]
	list_to_csv(csv_out, headers, final)
	print(csv_out)
	print(count)

def align():
	"""
	align a shap array;
	for the SHAP array:
		- every value represents 10 milliseconds (25 millisecond window)
	for the transcript array:
		- every value represents 1 second
		- has a speaker, np test, and words spoken
	- process transcript by:
		- create a NP array where the contents indicate the speaker and NP test.
		- each value represents one second.
	- align by:
		- iterate over SHAP array.
		- figure out which second timestamp we're in
			- divide each SHAP index by 100 to get the "second" conversion
				- (0 / 100 -> second=0)
					- at 0-25 milliseconds, we're b/w seconds (0, 1)
				- (50 / 100 -> seconds=0.5)
					- at 500-525 ms, we're b/w seconds (0, 1)
				* (100/100 -> seconds=1)
					- at 1000-1025 ms, we're b/w seconds (1, 2)
				* at (99/100 -> seconds=0.99)
					- at 990-1015 ms, we're technically mostly in (0,1), but partially in (1, 2)
	shap/5/2023_02_02/0_1/0-0127_20061108_[5]_vld_0_tst_1_neg.npy
	"""
	results_dir, vld_idx, tst_idx, cutoff, id_date_tscript, id_date_to_dur = config()
	tst_fps_data, csv_fn = set_tst_fps_data(vld_idx, tst_idx, results_dir, id_date_to_dur)
	count_and_write_csv(tst_fps_data, cutoff, results_dir, vld_idx, tst_idx, id_date_tscript,
		csv_fn)

def align_all():
	"""
	aligning all SHAP arrays
	"""
	results_dir, _, _, cutoff, id_date_tscript, id_date_to_dur = config()
	for vld_idx in range(5):
		for tst_idx in range(5):
			if vld_idx == tst_idx:
				continue
			if str(vld_idx) == 0 and str(tst_idx) == 1:
				continue
			## already did 0, 1
			tst_fps_data, csv_fn = set_tst_fps_data(vld_idx, tst_idx, results_dir, id_date_to_dur)
			count_and_write_csv(tst_fps_data, cutoff, results_dir, vld_idx, tst_idx, id_date_tscript,
				csv_fn)

def cp_data(src, dst):
	"""
	copy data
	"""
	command = f'cp "{src}" "{dst}"'
	with subprocess.Popen(command, shell=True) as process:
		process.communicate()
	print(f'copied {src} to {dst}')

def collect_all():
	"""
	collect all (unique) rows
	unique according to "duration_csv", not id_date
	copy over shap_map_fp and shap_fp to a different root folder
	add those as columns
	"""
	results_dir, *_ = config()
	csv_dir = os.path.join(results_dir, 'shap_csv')
	csv_list = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if\
		'shap_map_2023-03-28' in f and f.endswith('csv')]

	final_rows = defaultdict(list)
	for csv_in in csv_list:
		for row in yield_csv_data(csv_in):
			final_rows[row['duration_csv']].append(row)

	csv_out_dir = os.path.join(csv_dir, 'collect_all')
	shap_map_dir = os.path.join(csv_out_dir, 'shap_map')
	shap_fp_dir = os.path.join(csv_out_dir, 'shap_fp')

	for outdir in [csv_out_dir, shap_map_dir, shap_fp_dir]:
		if not os.path.isdir(outdir):
			os.makedirs(outdir)
	final_row_list = []
	for _, list_of_rows in final_rows.items():
		row = list_of_rows[0]
		shap_map_src = row['shap_map_fp']
		shap_map_fn = os.path.basename(shap_map_src)
		shap_map_dst = os.path.join(shap_map_dir, shap_map_fn)

		cp_data(shap_map_src, shap_map_dst)
		shap_fp_src = row['shap_fp']
		shap_fp_fn = os.path.basename(shap_fp_src)
		shap_fp_dst = os.path.join(shap_fp_dir, shap_fp_fn)
		cp_data(shap_fp_src, shap_fp_dst)

		row.update({'shap_map_dst': shap_map_dst, 'shap_fp_dst': shap_fp_dst})
		final_row_list.append(row)
	csv_fn = f'shap_collect_all_{get_time()}_[{len(final_row_list)}].csv'
	csv_out = os.path.join(csv_out_dir, csv_fn)

	headers = ['id_date', 'duration', 'pt_has_tscript', 'cutoff',
		'vld_idx', 'tst_idx', 'label', 'score', 'shap_map_dst', 'shap_fp_dst',
		'duration_csv', 'osm_fp', 'mfcc_fp',]
	list_to_csv(csv_out, headers, final_row_list, extrasaction='ignore')

def print_align():
	"""
	sample reading of collect_all_csv_in
	"""
	results_dir, *_ = config()
	collect_all_csv_in = os.path.join(results_dir, 'shap_csv', 'collect_all',
		'shap_collect_all_2023-03-28_161134.626191_[114].csv')
	for row in yield_csv_data(collect_all_csv_in):
		shap_map = np.load(row['shap_map_dst'], allow_pickle=True)
		shap_arr = np.load(row['shap_fp_dst'])
		shap_arr = np.swapaxes(np.squeeze(shap_arr), 0, 1)
		print(shap_map.shape)
		print(shap_arr.shape)
		all_data = []
		for idx, data in enumerate(shap_map):
			if np.any(data):
				speaker, seg_name = data
				shap_vals = shap_arr[idx]
				all_data.append((speaker, seg_name, shap_vals))

def ex_seg(seg_name_to_count, _):
	"""
	examine segment
	"""
	for seg_name, total_count in seg_name_to_count.items():
		print(seg_name)
		print(total_count * 10 / 1000)
		print()

def examine():
	"""
	examining distributions
	shap_csv/shap_map_2023-02-17_161639.663777_tst_audio_21639_vld_0_tst_1.csv
	"""
	results_dir, *_, = config()
	shap_csv = os.path.join(results_dir,
		'shap_csv/shap_map_2023-02-17_161639.663777_tst_audio_21639_vld_0_tst_1.csv')
	for row in yield_csv_data(shap_csv):
		non_pt = defaultdict(int)
		pt_only = defaultdict(int)
		shap_map = np.load(row['shap_map_fp'], allow_pickle=True)
		shap_arr = np.load(row['shap_fp'])
		shap_arr = np.swapaxes(np.squeeze(shap_arr), 0, 1)
		total_len = shap_arr.shape[0]
		for _, data in enumerate(shap_map):
			if np.any(data):
				speaker, seg_name = data
				if speaker.lower() != "p":
					non_pt[seg_name] += 1
				else:
					pt_only[seg_name] += 1
		print(f"label = {row['label']}")
		print('pt only')
		ex_seg(pt_only, total_len)
		print()
		print(f'total_pt: {sum(pt_only.values()) / 100}')
		print(f'total_non_pt: {sum(non_pt.values()) / 100}')
		print()

def plot_shap():
	"""
	plotting shap
	https://stackoverflow.com/questions/47585775/how-to-create-a-heatmap-of-a-single-dataframe-column
	plot tester speech only
	plot participant speech only
	plot both - use hatching to indicate speaker
	https://stackoverflow.com/questions/71855232/how-to-add-hatches-to-cells-in-seaborn-heatmap
	"""
	results_dir, *_ = config()
	shap_csv = os.path.join(results_dir,
		'shap_csv/shap_map_2023-02-17_161639.663777_tst_audio_21639_vld_0_tst_1.csv')
	osm_feats = ['F0final_sma', 'jitterLocal_sma', 'jitterDDP_sma', 'shimmerLocal_sma',
		'logHNR_sma']
	mfcc_feats = [f'mfcc{i}' for i in range(13)]
	feat_idx_to_name = dict(enumerate(osm_feats + mfcc_feats))
	for row in yield_csv_data(shap_csv):
		non_pt = defaultdict(int)
		pt_only = defaultdict(int)
		shap_map = np.load(row['shap_map_fp'], allow_pickle=True)
		shap_arr = np.load(row['shap_fp'])
		shap_arr = np.swapaxes(np.squeeze(shap_arr), 0, 1)
		total_len = shap_arr.shape[0]
		spk_identity_to_idx = defaultdict(list)
		feat_idx_to_shap_vals = defaultdict(list)
		for idx, data in enumerate(shap_map):
			for feat_idx, feat_shap_val in enumerate(shap_arr[idx]):
				feat_idx_to_shap_vals[feat_idx].append(feat_shap_val)
			if np.any(data):
				speaker, seg_name = data
				if speaker.lower() != "p":
					non_pt[seg_name] += 1
					spk_identity_to_idx['other_speaker'].append(idx)
				else:
					pt_only[seg_name] += 1
					spk_identity_to_idx['participant'].append(idx)
			else:
				spk_identity_to_idx['no_speaker'].append(idx)
		print(f"label = {row['label']}")
		print('pt only')
		ex_seg(pt_only, total_len)
		print()
		print(f'total_pt: {sum(pt_only.values()) / 100}')
		print(f'total_non_pt: {sum(non_pt.values()) / 100}')
		print()
		print(len(pt_only))
		print(len(non_pt))
		print(total_len)
		cog_label = 'DE' if str(row['label']) == '1' else 'CN'
		png_parent = f'{results_dir}/plots/{row["id_date"]}'
		if not os.path.isdir(png_parent):
			os.makedirs(png_parent)
		for feat_idx, list_of_shap_vals in feat_idx_to_shap_vals.items():
			print(f'graphing {feat_idx}')
			zscores = stats.zscore(list_of_shap_vals)
			vmin, vmax = min(zscores), max(zscores)
			_, axes = plt.subplots(1, 3)
			count = 0
			for speaker, list_of_indices in spk_identity_to_idx.items():
				current_shap_vals = [zscores[i] for i in list_of_indices]
				total_speech_time = len(current_shap_vals) * 10 / 1000
				name = f'{speaker}_{total_speech_time}s'
				feat_dataframe = pd.DataFrame({name: current_shap_vals})
				sns.heatmap(feat_dataframe, cmap='hot', yticklabels=False,
					vmin=vmin, vmax=vmax, ax=axes[count])
				count += 1
				plt.tight_layout()
			png_fn = f'{cog_label}_{feat_idx_to_name[feat_idx]}.png'
			png_fp = os.path.join(png_parent, png_fn)
			plt.savefig(png_fp, dpi=400)
			print(png_fp)
			plt.close()

if __name__ == '__main__':
	print_align()
