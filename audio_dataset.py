"""
audio_dataset.py
AudioDataset();
meant for loading audio data;
"""
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import fhs_split_dataframe as sdf
from get_vols import get_vols

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

		data_headers = kwargs.get('data_headers', ['patient_id', 'mfcc_fp',
			'start', 'end'])

		idx_to_region = kwargs.get('idx_to_region')
		ordered_idx = dict(sorted(idx_to_region.items(), key=lambda item: item[0]))
		normalize_fsl = kwargs.get('normalize_fsl', False)
		if not normalize_fsl:
			regions = list(ordered_idx.values())
		else:
			regions = []
			for _, region in ordered_idx.items():
				regions.append(f'{region}_brain_frac')
		data_headers.extend(regions)
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
			row_data = [pid, row['mfcc_fp'], None, None]
			row_data.extend([row[r] for r in regions])
			data_list.append(row_data)

		self.df_dat = pd.DataFrame(data_list, columns=data_headers)
		self.patient_list = list(set(current_fold_ids))
		self.patient_list.sort()
		self.ordered_idx = ordered_idx
		self.normalize_fsl = normalize_fsl

	def __len__(self):
		"""
		get length
		"""
		return len(self.df_dat)

	def __getitem__(self, idx):
		"""
		get item;
		"""
		audio_fp = self.df_dat.loc[idx, 'mfcc_fp']
		fea = np.load(audio_fp)
		start = self.df_dat.loc[idx, 'start']
		end = self.df_dat.loc[idx, 'end']
		if (start is not None and end is not None) and\
			(not np.isnan(start) and not np.isnan(end)):
			start = int(start)
			end = int(end)
			fea = fea[start:end]
		target = get_vols(self.ordered_idx, self.df_dat, idx, self.normalize_fsl)
		return fea, target, self.df_dat.loc[idx, 'patient_id'], audio_fp, start, end

def collate_fn(batch):
	"""
	collect audio path, label, patient ID
	"""
	aud = [itm[0] for itm in batch]
	target = np.stack([itm[1] for itm in batch])
	audio_filepaths = np.stack([itm[3] for itm in batch])
	start_end = [(itm[4], itm[5]) for itm in batch]
	return aud, target, audio_filepaths, start_end
