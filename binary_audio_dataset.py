"""
audio_dataset.py
AudioDataset();
meant for loading audio data;
"""
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import fhs_split_dataframe as sdf

class BinaryAudioDataset(Dataset):
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

		self.audio_idx = kwargs.get('audio_idx')
		self.get_label = kwargs.get('get_label')
		self.get_fea = kwargs.get('get_fea', np.load)
		self.get_fea_kw = kwargs.get('get_fea_kw', {})
		data_headers = kwargs.get('data_headers', ['patient_id', self.audio_idx, 'label',
			'start', 'end'])

		assert mode in ['TRN', 'VLD', 'TST'], mode
		self.mode = mode
		df_raw = pd.read_csv(csv_info, dtype=object)
		if tst_idx is not None:
			pt_ids = get_pt_ids(df_raw, **get_pt_ids_kw)
			## get all participant IDs;
			folds = sdf.create_folds(pt_ids, num_folds, seed)
			current_fold_ids = set(sdf.get_fold(pt_ids, folds, vld_idx, tst_idx, mode))
		else:
			static_ids, other_ids = sdf.get_static_and_remaining_ids(df_raw, sdf.has_transcript,
				get_pt_ids)
			if mode == 'TST':
				current_fold_ids = static_ids
			else:
				folds = sdf.create_folds(other_ids, num_folds, seed)
				current_fold_ids = set(sdf.get_holdout_fold(other_ids, folds, vld_idx, mode))
		data_list = []
		for _, row in df_raw.iterrows():
			pid = get_pid(row, **get_pid_kw)
			if pid not in current_fold_ids:
				continue
			label = self.get_label(row)
			if self.audio_idx == 'osm_and_mfcc_npy':
				filepath = (row['osm_npy'], row['mfcc_npy'])
			else:
				filepath = row[self.audio_idx]
			row_data = [pid, filepath, label, None, None]
			data_list.append(row_data)

		self.df_dat = pd.DataFrame(data_list, columns=data_headers)
		self.labels = self.df_dat.label.to_numpy()
		self.patient_list = list(set(current_fold_ids))
		self.patient_list.sort()
		self.num_patients = len(self.patient_list)
		self.num_audio = len(data_list)
		self.num_positive_audio = sum([n for _, _, n, *_ in data_list])
		self.num_negative_audio = self.num_audio - self.num_positive_audio

	def __len__(self):
		"""
		get length
		"""
		return len(self.df_dat)

	def __getitem__(self, idx):
		"""
		get item;
		"""
		audio_fp = self.df_dat.loc[idx, self.audio_idx]
		fea = self.get_fea(audio_fp, **self.get_fea_kw)
		start = self.df_dat.loc[idx, 'start']
		end = self.df_dat.loc[idx, 'end']
		if (start is not None and end is not None) and\
			(not np.isnan(start) and not np.isnan(end)):
			start = int(start)
			end = int(end)
			fea = fea[start:end]
		label = self.df_dat.loc[idx, 'label']
		return fea, label, audio_fp, start, end


def collate_fn(batch):
	"""
	collect audio path, label, patient ID
	"""
	aud = [itm[0] for itm in batch]
	target = np.stack([itm[1] for itm in batch])
	audio_filepaths = np.stack([itm[2] for itm in batch])
	start_end = [(itm[3], itm[4]) for itm in batch]
	return aud, target, audio_filepaths, start_end
