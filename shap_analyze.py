"""
shap_analyze.py

select dataset;
analyze SHAP;
"""
from collections import defaultdict
import os
import re
import json
import shap
import torch
import numpy as np
from csv_read import yield_csv_data
from tcn import TCN
from load_all_data import shap_load_osm_mfcc

def json_load(filename):
	"""
	open a json-like file object
	"""
	with open(filename, 'r') as file:
		final = json.load(file)
	return final

def get_tst_pids(csv_in):
	"""
	read in tst csv and get set of patient_ids (FHS IDs)
	patient_id, osm_and_mfcc_npy, label, start, end, score;
	"""
	return {r['patient_id'] for r in yield_csv_data(csv_in)}

def select_data(csv_in, json_in):
	"""
	select those with certain duration cutoffs + tscripts;
	"""
	durations = []
	pt_ids = get_tst_pids(csv_in)
	tscript_durs = []
	for pt_id, list_of_data in json_load(json_in).items():
		for data in list_of_data:
			if pt_id in pt_ids:
				durations.append(data['duration'])
				if data['pt_has_tscript'] == 1:
					tscript_durs.append(data['duration'])
	cutoffs = [5, 15, 30, 45, 60]
	count = defaultdict(int)
	for duration in durations:
		for cutoff in cutoffs:
			if duration >= cutoff:
				count[cutoff] += 1
	tscript_count = defaultdict(int)
	for dur in tscript_durs:
		for cutoff in cutoffs:
			if dur >= cutoff:
				tscript_count[cutoff] += 1
	print(f'total_recordings: {len(durations)}')
	print(f'total_recordings_with_tscripts: {len(tscript_durs)}')
	print()
	for cutoff, total in count.items():
		print(f'cutoff: {cutoff}')
		print(f'total: {total}')
		print(f'tscript_total: {tscript_count[cutoff]}')
		print()

def test_count():
	"""
	test counting durations;
	"""
	csv_in = "results/HC_vs_DE_tscript_osm_and_mfcc_npy_1.0_1.0_lr_nworkers0_orig_tcn/"+\
		"32_epochs/21639/"+\
		"tst_audio_21639_vld_0_tst_1.csv"
	json_in = "json_in/"+\
		"remap_(760)_[1511]_20221021_14_13_54_0011_osm_dr_longest_mfccs_osm_npy_tscript.json"
	select_data(csv_in, json_in)

def get_id_date_to_dur(json_in):
	"""
	map id_dates to duration and pt_has_tscript;
	"""
	final = {}
	for _, list_of_data in json_load(json_in).items():
		for data in list_of_data:
			id_date = data['id_date']
			assert id_date not in final, id_date
			final[id_date] = {'duration': data['duration'],
				'pt_has_tscript': data['pt_has_tscript']}
	return final

def get_tst_fps_data(csv_in, id_date_to_dur, limit=None, **kwargs):
	"""
	get osm and mfcc filepaths from tst csv;
	attach duration and pt_has_tscript
	"""
	add_keys = kwargs.get('add_keys', [])
	final = {}
	ptrn = re.compile(r"/(\d{1,2}-\d{4}_\d{8})/")
	count = 0
	for row in yield_csv_data(csv_in):
		if limit is not None and limit <= count:
			break
		count += 1
		osm_and_mfcc_npy = row['osm_and_mfcc_npy']
		osm_fp, mfcc_fp = osm_and_mfcc_npy.split(',')
		osm_fp = osm_fp.replace('(', '').replace("'", "")
		mfcc_fp = mfcc_fp.replace(')', '').replace("'", "").replace(" ", "")
		assert os.path.isfile(osm_fp), osm_fp
		assert os.path.isfile(mfcc_fp), mfcc_fp
		osm_id_date = ptrn.search(osm_fp).groups()[0]
		mfcc_id_date = ptrn.search(mfcc_fp).groups()[0]
		assert osm_id_date == mfcc_id_date, osm_fp
		assert osm_id_date not in final, osm_id_date
		dur_tscript = id_date_to_dur[osm_id_date]
		final[osm_id_date] = {'osm_fp': osm_fp, 'mfcc_fp': mfcc_fp,
			'duration': dur_tscript['duration'],
			'pt_has_tscript': dur_tscript['pt_has_tscript']}
		final[osm_id_date].update({k: row[k] for k in add_keys})
	return final

def select_targets(fp_dict, full_array, cutoff):
	"""
	select those with appropriate durations and those with
	tscripts;

	fp_dict[fp_tuple] = {final_arr, duration, pt_has_tscript}
	"""
	targets = {d['id_date']: full_array[d['idx']] for d in fp_dict.values()\
		if float(d['duration']) >= cutoff\
		and int(d['pt_has_tscript']) == 1}
	return targets

def main():
	"""
	main entrypoint;
	"""
	json_in = "json_in/"+\
		"remap_(760)_[1511]_20221021_14_13_54_0011_osm_dr_longest_mfccs_osm_npy_tscript.json"
	device = "cuda:1"
	seed = 21639
	# cutoff = 45
	# cutoff = 10
	cutoff = 5
	results_dir = "results/HC_vs_DE_tscript_osm_and_mfcc_npy_1.0_1.0_lr_nworkers0_orig_tcn/"+\
		f"32_epochs/{seed}/"
	today = "2023_02_02"
	# vld_idx, tst_idx = 0, 1
	# vld_idx, tst_idx = 0, 2
	for vld_idx in range(5):
		for tst_idx in range(5):
			if vld_idx == tst_idx:
				continue
			# if vld_idx == 0 and tst_idx == 1:
			# 	continue ## already done on 12-12
			# if vld_idx == 0 and tst_idx == 2:
			# 	continue ## already done on 12-12
			print(f'starting vld ({vld_idx}), tst ({tst_idx})')
			ckpt_fp = f"{results_dir}/pt_files/vld_{vld_idx}_tst_{tst_idx}.pt"
			csv_in = f"{results_dir}/tst_audio_{seed}_vld_{vld_idx}_tst_{tst_idx}.csv"
			ckpt = torch.load(ckpt_fp)
			ckpt = ckpt.to(device)

			net = TCN(ys_len=2,channels=18,device=device)
			net.load_state_dict(ckpt.state_dict())
			net.eval()
			# print(net)

			id_date_to_dur = get_id_date_to_dur(json_in)
			tst_fps_data = get_tst_fps_data(csv_in, id_date_to_dur, limit=None)
			fp_dict, full_array = shap_load_osm_mfcc(tst_fps_data, cutoff)
			targets = select_targets(fp_dict, full_array, cutoff)
			print(len(fp_dict))
			print(full_array.shape)
			print(len(targets))

			full_array = torch.from_numpy(full_array).type(torch.FloatTensor).to(device)
			## batch x 270000 x 18
			full_array = torch.unsqueeze(full_array, 1)
			## batch x 1 x 270000 x 18
			full_array = full_array.swapaxes(2, 3)
			## convert to batch x 1 x 18 x 270000
			try:
				shap_baseline = shap.GradientExplainer(net, full_array)
				print(shap_baseline)
			except RuntimeError as error:
				print(error)
				continue ## likely CUDA memory issue
			for id_date, target in targets.items():
				## target shape is length x 18
				target = torch.from_numpy(target).type(torch.FloatTensor).to(device)
				target = torch.unsqueeze(target, 0)
				target = torch.unsqueeze(target, 0)
				## target shape is 1 x 1 x length x 18
				target = target.swapaxes(2, 3)
				## 1 x 1 x 18 x length
				shap_val = shap_baseline.shap_values(target)
				neg_shap, pos_shap = shap_val
				print(neg_shap.shape)
				print(pos_shap.shape)
				shap_parent = f'{results_dir}/shap/{cutoff}/{today}/{vld_idx}_{tst_idx}/'
				neg_fp = f'{shap_parent}/{id_date}_[{cutoff}]_vld_{vld_idx}_tst_{tst_idx}_neg.npy'
				pos_fp = f'{shap_parent}/{id_date}_[{cutoff}]_vld_{vld_idx}_tst_{tst_idx}_pos.npy'
				if not os.path.isdir(os.path.dirname(neg_fp)):
					os.makedirs(os.path.dirname(neg_fp))
				np.save(neg_fp, neg_shap)
				np.save(pos_fp, pos_shap)
				print(neg_fp)
				print(pos_fp)

if __name__ == '__main__':
	main()
