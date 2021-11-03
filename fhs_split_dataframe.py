"""
fhs_split_dataframe.py
module for creating folds, splitting datasets;
"""
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
