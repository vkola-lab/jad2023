"""
random_forest.py
running a random forest on age, sex, education, etc;
"""
import os
import pprint
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

from misc import populate_met, get_time, get_date
import misc as msc
from plot_directory import plot_individual_curve

FEAT_COLS = ['age', 'encoded_sex', 'edu']
# FEAT_COLS = ['age', 'encoded_sex']
# FEAT_COLS = ['age']
EXT = '_'.join(FEAT_COLS)
if len(FEAT_COLS) == 1:
	PLOT_EXT = {'age': 'Age', 'encoded_sex': 'Sex', 'edu': 'Education'}[FEAT_COLS[0]]
else:
	PLOT_EXT = 'Demographics'


def sample():
	"""
	run a sample
	"""
	feats = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
	labels = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
	groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
	gkf = GroupKFold(n_splits=3)
	for train, test in gkf.split(feats, labels, groups=groups):
		print(type(train))
		print(type(test))
		print(train)
		print(test)
		print(groups[train])

def get_feats_labels_groups(df, feat_cols, label_idx, group_idx):
	"""
	get feats, labels, and groups
	"""
	feats = df[feat_cols]
	feats = feats.fillna(-1)
	## a few missing education values replaced by -1 instead of NaN
	labels = df[label_idx]
	groups = df[group_idx]
	return feats, labels, groups

def get_data():
	"""
	get data from csv
	"""
	# csv_in = "csv_in/nde_vs_de/"+\
	# 	"remap_(760)_[1511]_20230714_9_25_34_0488_voice_and_funcs_and_demo_NDE_vs_DE.csv"
	csv_in = "csv_in/edu/"+\
		"remap_(760)_[1511]_20230724_15_4_2_0772_voice_funcs_demo_add_edu_NDE_vs_DE.csv"
	df = pd.read_csv(csv_in)
	df['fid'] = df['idtype'].astype(str) + '-' + df['id'].astype(str)
	# df['fid'] = df.apply(lambda x: str(x['idtype']) + '-' + str(df['id']).zfill(4), axis=1)
	hc_df = df[df['is_norm'] == 1]
	mci_df = df[df['is_mci'] == 1]
	de_df = df[df['is_demented'] == 1]


	label_idx = 'is_demented'
	group_idx = 'fid'
	hc_data = get_feats_labels_groups(hc_df, FEAT_COLS, label_idx,
		group_idx)
	mci_data = get_feats_labels_groups(mci_df, FEAT_COLS, label_idx,
		group_idx)
	de_data = get_feats_labels_groups(de_df, FEAT_COLS, label_idx,
		group_idx)
	return hc_data, mci_data, de_data

def run_rf(feats, labels, groups, ext):
	"""
	run the random forest
	"""
	rf = RandomForestClassifier()
	gkf = GroupKFold(n_splits=5)
	avg_stats = defaultdict(list)
	all_labels = []
	all_probs = []
	for idx, (train, test) in enumerate(gkf.split(feats, labels, groups=groups)):
		print(f'Fold: {idx}')
		x_train = feats.iloc[train]
		y_train = labels.iloc[train]
		rf.fit(x_train, y_train)

		x_test = feats.iloc[test]
		y_test = labels.iloc[test]
		y_pred = rf.predict(x_test)
		y_pred_prob = rf.predict_proba(x_test)[:, 1]
		all_labels.append(y_test)
		all_probs.append(y_pred_prob)
		met = {}
		mat = confusion_matrix(y_test, y_pred)
		true_neg, false_pos, false_neg, true_pos = mat.ravel()
		populate_met(met, true_neg, true_pos, false_neg, false_pos, y_test, y_pred, y_pred_prob)
		pprint.pprint(met)
		for met, val in met.items():
			avg_stats[met].append(val)
	today = get_date()
	cur_time = get_time()

	parent_dir = f'random_forest/{today}/{cur_time}/{ext}'
	if not os.path.isdir(parent_dir):
		os.makedirs(parent_dir)

	plot_stats(all_labels, all_probs, ext, parent_dir)
	feat_imp_fig_name = os.path.join(parent_dir, f'feat_imp_{EXT}.svg')
	importances = plot_feat_importance(rf, feat_imp_fig_name)
	stats_to_txt(avg_stats, ext, parent_dir, importances)

def stats_to_txt(avg_stats, ext, parent_dir, importances):
	"""
	write stats to txt
	"""
	importances = dict(zip(FEAT_COLS, importances))
	txt_fp = os.path.join(parent_dir, f'{ext}_metrics.txt')
	lines = []
	with open(txt_fp, 'w') as outfile:
		lines.append('avg_performance:\n')
		for met_name, list_of_vals in avg_stats.items():
			mean = np.mean(list_of_vals)
			std = np.std(list_of_vals)
			lines.append(f'{met_name}: {mean:.3f}, {std:.3f}\n')
		lines.append('\nfeat_importance:\n')
		for feat_name, importance_val in importances.items():
			lines.append(f'{feat_name}: {importance_val} \n')
		outfile.write(''.join(lines))
		print(''.join(lines))
	print(txt_fp)

def plot_stats(y_test, y_pred_prob, ext, parent_dir):
	"""
	plotting stats
	"""
	curr_hmp_roc = msc.get_roc_info(y_test, y_pred_prob)
	curr_hmp_pr  = msc.get_pr_info(y_test, y_pred_prob)
	# legend_dict = {0: ('magenta', 'Age')}
	legend_dict = {0: ('magenta', PLOT_EXT)}
	fig_name = os.path.join(parent_dir, f'{ext}_roc.svg')
	plot_individual_curve(curr_hmp_roc, legend_dict, 'roc', fig_name)
	print(fig_name)
	fig_name = os.path.join(parent_dir, f'{ext}_pr.svg')
	plot_individual_curve(curr_hmp_pr, legend_dict, 'pr', fig_name)
	print(fig_name)

def plot_feat_importance(rf, fig_name):
	"""
	plot importance of features
	"""
	importances = rf.feature_importances_
	std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
	forest_importances = pd.Series(importances, index=FEAT_COLS)
	fig, ax = plt.subplots()
	forest_importances.plot.bar(yerr=std, ax=ax)
	ax.set_title("Feature importances using MDI")
	ax.set_ylabel("Mean decrease in impurity")
	fig.tight_layout()
	fig.savefig(fig_name, dpi=300, format='svg')
	plt.close('all')
	print(importances)
	print(std)
	print(fig_name)
	return importances

def hc_vs_de():
	"""
	run hc vs DE
	"""
	hc_data, _, de_data = get_data()
	hc_feats, hc_labels, hc_groups = hc_data
	de_feats, de_labels, de_groups = de_data
	feats = hc_feats.append(de_feats)
	labels = hc_labels.append(de_labels)
	groups = hc_groups.append(de_groups)
	ext = f'HC_vs_DE_{EXT}'
	run_rf(feats, labels, groups, ext)

def mci_vs_de():
	"""
	run mci vs de
	"""
	_, mci_data, de_data = get_data()
	mci_feats, mci_labels, mci_groups = mci_data
	de_feats, de_labels, de_groups = de_data
	feats = mci_feats.append(de_feats)
	labels = mci_labels.append(de_labels)
	groups = mci_groups.append(de_groups)
	ext = f'MCI_vs_DE_{EXT}'
	run_rf(feats, labels, groups, ext)

def nde_vs_de():
	"""
	run nde vs de
	"""
	hc_data, mci_data, de_data = get_data()
	hc_feats, hc_labels, hc_groups = hc_data
	mci_feats, mci_labels, mci_groups = mci_data
	de_feats, de_labels, de_groups = de_data
	feats = hc_feats.append(mci_feats).append(de_feats)
	labels = hc_labels.append(mci_labels).append(de_labels)
	groups = hc_groups.append(mci_groups).append(de_groups)
	ext = f'NDE_vs_DE_{EXT}'
	run_rf(feats, labels, groups, ext)

if __name__ == '__main__':
	# hc_vs_de()
	mci_vs_de()
