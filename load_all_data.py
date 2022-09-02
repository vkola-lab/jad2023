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
