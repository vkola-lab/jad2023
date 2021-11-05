"""
audio_dataset.py
AudioDataset();
meant for loading audio data;
"""
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import fhs_split_dataframe as sdf

class AudioDataset(Dataset):
	"""
	audio dataset;
	"""
	def __init__(self, csv_info, mode, **kwargs):
		"""
		initializing func;
		"""
		num_folds = kwargs.get('num_folds', 5)
		vld_idx = kwargs.get('vld_idx')
		tst_idx = kwargs.get('tst_idx')
		seed = kwargs.get('seed')

		get_pt_ids = kwargs.get('get_pt_ids', sdf.get_fhs_ids)
		get_pt_ids_kw = kwargs.get('get_pt_ids_kw', {})

		get_pid = kwargs.get('get_pid', lambda r: f'{r.idtype}-{r.id.zfill(4)}')
		get_pid_kw = kwargs.get('get_pid_kw', {})

		yield_data_and_target = kwargs.get('yield_data_and_target', sdf.yield_aud_and_mni)
		yield_data_and_target_kw = kwargs.get('yield_data_and_target_kw', {})

		data_headers = kwargs.get('data_headers', ['patient_id', 'audio_fn', 'mni_brain'])

		self.mni_fp_to_vector = kwargs.get('mni_fp_to_vector')

		assert mode in ['TRN', 'VLD', 'TST'], mode
		self.mode = mode
		df_raw = pd.read_csv(csv_info, dtype=object)
		pt_ids = get_pt_ids(df_raw, **get_pt_ids_kw)
		## get all participant IDs;
		folds = sdf.create_folds(pt_ids, num_folds, seed)
		current_fold_ids = set(sdf.get_fold(pt_ids, folds, vld_idx, tst_idx, mode))
		data_list = []
		for _, row in df_raw.iterrows():
			pid = get_pid(row, **get_pid_kw)
			if pid not in current_fold_ids:
				continue
			for data, target in yield_data_and_target(row, **yield_data_and_target_kw):
				data_list.append([pid, data, target])

		self.df_dat = pd.DataFrame(data_list, columns=data_headers)
		self.targets = self.get_targets()
		self.patient_list = list(set(current_fold_ids))
		self.patient_list.sort()

	def __len__(self):
		"""
		get length
		"""
		return len(self.df_dat)

	def __getitem__(self, idx):
		"""
		get item;
		"""
		audio_fp = self.df_dat.loc[idx, 'audio_fn']
		fea = np.load(audio_fp)
		mni_brain = self.df_dat.loc[idx, 'mni_brain']
		target = self.mni_fp_to_vector[mni_brain]
		return fea, target, self.df_dat.loc[idx, 'patient_id'], audio_fp

	def get_targets(self):
		"""
		convert target column to np array;
		"""
		return np.array([self.mni_fp_to_vector[row['mni_brain']]\
			for _, row in self.df_dat.iterrows()])

def collate_fn(batch):
	"""
	collect audio path, label, patient ID
	"""
	aud = [itm[0] for itm in batch]
	target = np.concatenate([itm[1] for itm in batch], axis=0)
	audio_filepaths = np.stack([itm[3] for itm in batch])
	return aud, target, audio_filepaths
