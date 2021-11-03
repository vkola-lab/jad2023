"""
audio_dataset.py
AudioDataset();
meant for loading audio data;
"""
from torch.utils.data import Dataset
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

		assert mode in ['TRN', 'VLD', 'TST'], mode
		self.mode = mode
		self.dataframe = None
		df_raw = pd.read_csv(csv_info, dtype=object)
		pt_ids = get_pt_ids(df_raw, **get_pt_ids_kw)
		## get all participant IDs;
		folds = sdf.create_folds(pt_ids, num_folds, seed)
		fold = sdf.get_fold(pt_ids, folds, vld_idx, tst_idx, mode)
		print(mode)
		print(fold)
		print()
