"""
get_vols.py
getting volume array from csv
"""
import re
import numpy as np

def map_idx_to_region(atlas_xml):
	"""
	read atlas_xml as a txt file;
	parse it for the index to region name mapping;
	"""
	ptrn = r"<label><index>(\d+)</index><name>(.*)</name></label>"
	compiled = re.compile(ptrn)
	idx_to_region = {0: 'background'}
	with open(atlas_xml, 'r') as infile:
		for line in infile:
			search = compiled.search(line)
			if search is not None:
				idx, region = search.groups()
				idx_to_region[int(idx)] = region
	return idx_to_region

def get_vols(ordered_idx, df_dat, idx, normalize_fsl):
	"""
	get raw volumes or brain fracs;
	"""
	vol_list = []
	for _, region in ordered_idx.items():
		row_idx = region if not normalize_fsl else f'{region}_brain_frac'
		vol_list.append(float(df_dat.loc[idx, row_idx]))
	return np.array(vol_list)
