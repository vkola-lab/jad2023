"""
fhs_split_dataframe.py
module for creating folds, splitting datasets;
"""
import os
import random
import numpy as np

def get_fhs_ids(df_pts):
	"""
	from a dataframe, get idtype+id (forms a unique FHS ID);
	return an array with all the unique FHS IDs;
	"""
	idtypes = df_pts.idtype.values.ravel('K')
	ids = df_pts.id.values.ravel('K')
	return np.unique([f'{idtypes[i]}-{str(ids[i]).zfill(4)}' for i, _ in enumerate(idtypes)])

def create_folds(sample_ids, num_folds, seed):
	"""
	take datasamples, split them into a number of folds (num_folds), set random seed;
	"""
	random.seed(seed)
	lst_idx = np.array(range(len(sample_ids)))
	random.shuffle(lst_idx)
	return [lst_idx[np.arange(len(sample_ids)) % num_folds == i] for i in range(num_folds)]

def get_fold(sample_ids, folds, vld_idx, tst_idx, mode):
	"""
	fld: numpy array containing the folds and the data indices for that fofld;
	vld_idx: validation fold index;
	tst_idx: test fold index;
	mode: 'VLD', 'TST', 'TRN'
	"""
	assert mode in {'TRN', 'VLD', 'TST'}, f"{mode} is not TRN VLD OR TST"
	if mode == 'VLD':
		idx = folds[vld_idx]
	elif mode == 'TST':
		idx = folds[tst_idx]
	elif mode == 'TRN':
		all_fold_indices = np.arange(len(folds))
		## if 5 folds, then all_fold_indices = [0, 1, 2, 3, 4]
		all_fold_indices = all_fold_indices[all_fold_indices != vld_idx]
		all_fold_indices = all_fold_indices[all_fold_indices != tst_idx]
		## keep all fold indices except for the TRN and VLD indices;
		idx = np.concatenate([folds[all_fold_indices[i]] for i in range(len(all_fold_indices))])
	return sample_ids[idx]

def yield_aud_and_mni(row):
	"""
	yield audio fn and mni vector;
	"""
	yield (row['seg_fp'], row['mni_brain'])

def yield_rand_seg_and_mni(row, **kwargs):
	"""
	yield <num_pt_segments> random 5 minute segments of pt speech and mni vector
	"""
	mni_vector = row['mni_brain']

	num_pt_segments = kwargs.get('num_pt_segments')
	seg_min = kwargs.get('seg_min', 5)

	pt_only_fp = row['pt_npy']
	pt_only_npy = np.load(pt_only_fp)
	seg_dur = seg_min * 60
	assert pt_only_npy.shape[0] % 100 == 0, row['pt_npy']
	last_start = int(pt_only_npy.shape[0] / 100 - seg_dur)
	## the latest start we can pick is the length of the audio minus length of segment
	## length of audio in seconds is shape / 100 bc each MFCC unit represents 10 milliseconds
	## length of segment is in seconds already;
	all_pairs = [(s, s + seg_dur) for s in range(last_start + 1)]
	chosen_pairs = set()
	for _ in range(num_pt_segments):
		pair = random.choice(all_pairs)
		while pair in chosen_pairs:
			pair = random.choice(all_pairs)
		chosen_pairs.add(pair)
		start, end = pair
		start *= 100
		end *= 100
		## convert timestamp<seconds> to timestamp<10-milliseconds> for MFCC indexing
		yield (pt_only_fp, mni_vector, start, end)

def create_pt_segment(row, pt_segment_root, pt_only_npy, start, end):
	"""
	create pt segment npy;
	"""
	id_date = row['id_date']
	npy_dir = os.path.join(pt_segment_root, id_date.split('_')[0], id_date)
	npy_fn = f'start_{start}_end_{end}_{id_date}.npy'
	npy_fp = os.path.join(npy_dir, npy_fn)
	if not os.path.isfile(npy_fp):
		if not os.path.isdir(npy_dir):
			os.makedirs(npy_dir)
		pt_segment = pt_only_npy[start*100:end*100]
		## convert timestamp<seconds> to timestamp<10-milliseconds> for MFCC indexing
		np.save(npy_fp, pt_segment)
		print(f'created {npy_fp};')
	return npy_fp
