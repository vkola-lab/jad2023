"""
stratified_load.py
load stratified folds;
"""
from sklearn.model_selection import StratifiedShuffleSplit

def stratified_load(csv_in, seed):
	"""
	load stratified folds;
	"""
	df_raw = pd.read_csv(csv_in, dtype=object)
	pt_ids = []
	