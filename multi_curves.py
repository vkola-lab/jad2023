#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 18:17:08 2020

@author: cxue2
"""
import os
import sys
import glob
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from misc import calc_performance_metrics
from misc import get_roc_info, get_pr_info, get_date

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['savefig.facecolor'] = 'w'

def plot_curve(curve, ax, xs, ys_mean, ys_upper, ys_lower, auc_mean,
	auc_std, color='C0'):
	"""
	plot curve;
	"""
	assert curve in ['roc', 'pr']
	if curve == 'roc':
		# ys_mean = ys_mean[::-1]
		# ys_upper = ys_upper[::-1]
		# ys_lower = ys_lower[::-1]
		xlabel, ylabel = '1 - Specificity', 'Sensitivity'
		# xs = 1 - xs
	else:
		xlabel, ylabel = 'Recall', 'Precision'

	p_mean, = ax.plot(
		xs, ys_mean, color=color,
		linestyle='-',
		lw=1.5, alpha=1)

	p_fill = ax.fill_between(
		xs, ys_lower, ys_upper,
		alpha=.4,
		facecolor='none',
		edgecolor=color,
		hatch='//////')

	ax.set_xlim([-0.05, 1.05])
	ax.set_ylim([-0.05, 1.05])
	ax.set_xlabel(xlabel, fontweight='bold')
	ax.xaxis.set_label_coords(0.5, -0.01)
	ax.set_ylabel(ylabel, fontweight='bold')
	ax.yaxis.set_label_coords(-0.01, 0.5)
	ax.set_title('', fontweight='bold')
	ax.set_xticks([0, 1])
	ax.set_xticklabels(ax.get_xticks(), weight='bold')
	ax.set_yticks([0, 1])
	ax.set_yticklabels(ax.get_xticks(), weight='bold')
	ax.set_aspect('equal', 'box')
	ax.set_facecolor('w')
	plt.setp(ax.spines.values(), color='w')
	# ax.axhline(0.9, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
	# ax.axhline(0.8, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
	# ax.axvline(0.9, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
	# ax.axvline(0.8, linestyle='-', color='#CCCCCC', lw=1, zorder=0)
	ax.axvline(0.0, linestyle='-', color='k', lw=1, zorder=1)
	ax.axhline(0.0, linestyle='-', color='k', lw=1, zorder=1)
	return p_mean, p_fill, auc_mean, auc_std

def plot_curves(data, legend_dict, curve_str, fig_name):
	"""
	plot all the curves;
	"""
	fig, ax = plt.subplots(1)
	legend_str = {}
	for idx, hmp_roc in data.items():
		color, legend_ext = legend_dict[idx]
		p_mean, _, auc_mean, auc_std = plot_curve(curve_str, ax, hmp_roc['xs'],
			hmp_roc['ys_mean'], hmp_roc['ys_upper'],
			hmp_roc['ys_lower'], hmp_roc['auc_mean'], hmp_roc['auc_std'],
			color=color)
		msg = r'{}: {:.3f}$\pm${:.3f}'.format(legend_ext, auc_mean, auc_std)
		legend_str[idx] = (p_mean, msg)
	p_mean_list = [v[0] for k, v in legend_str.items()]
	msg_list = [v[1] for k, v in legend_str.items()]
	ax.legend(p_mean_list, msg_list,
			  facecolor='w', prop={"weight":'bold', "size":17},
			  bbox_to_anchor=(0.04, 0.04, 0.5, 0.5),
			  loc='lower left')

	fig.savefig(fig_name, dpi=300, format='svg')
	print(fig_name)

def main():
	"""
	main entrypoint
	"""
	cnn_dir_rsl = sys.argv[1]
	lstm_dir_rsl = sys.argv[2]
	first_ext, second_ext = 'MFCC', 'OSM'
	tiff_dir = os.path.join('tiff', get_date())
	if len(sys.argv) > 3:
		print(sys.argv)
		first_ext, second_ext = sys.argv[3], sys.argv[4]
	if len(sys.argv) == 6:
		tiff_dir = os.path.join(tiff_dir, sys.argv[5])
	if not os.path.isdir(tiff_dir):
		os.makedirs(tiff_dir)
	roc_dict = {}
	pr_dict = {}
	for idx, dir_rsl in enumerate([cnn_dir_rsl, lstm_dir_rsl]):
		mode = 'audio_avg'
		audio_idx = 'mfcc_npy' if idx == 0 else 'osm_npy'
		# list of all csv files
		num_csvs = None
		if num_csvs is None:
			lst_csv = glob.glob(dir_rsl + '/*.csv', recursive=False)
			dirs_read = [dir_rsl]
		else:
			lst_csv = []
			dirs_read = []
			directories = [os.path.join(dir_rsl, d) for d in os.listdir(dir_rsl)]
			directories = [d for d in directories if os.path.isdir(d)]
			for directory in directories:
				current_lst = glob.glob(directory + '/*.csv', recursive=False)
				if len(current_lst) == int(num_csvs):
					lst_csv.extend(current_lst)
					dirs_read.append(directory)
		lst_lbl, lst_scr = [], []
		mtr_all = defaultdict(list)
		assert lst_csv != [], f'dirs_read: {dirs_read}, {os.listdir(dirs_read[0])}'
		print(f"{len(lst_csv)} csvs found;")
		print("\n".join(dirs_read))
		fn_metrics = {}
		for fn in lst_csv:
			fn_base = os.path.basename(fn)
			if not fn_base.startswith('tst_audio'):
				continue
			# read from csv
			df = pd.read_csv(fn)
			# get scores and labels
			if mode == 'chunk':
				lbl = df.label.to_numpy()
				scr = df.score.to_numpy()
			elif mode == 'audio_avg':
				tmp = df.groupby(audio_idx).mean().to_numpy()
				lbl = tmp[:,0].astype(np.int)
				scr = tmp[:,-1]
			mtr = calc_performance_metrics(scr, lbl)
			for k, _ in mtr.items():
				if k == 'mat':
					continue
				mtr_all[k].append(mtr[k])
			fn_metrics[fn] = {mk: mv for mk, mv in mtr.items() if mk != 'mat'}
			lst_lbl.append(lbl)
			lst_scr.append(scr)
		for filename, fn_mtr in fn_metrics.items():
			print(filename)
			for metric, metric_val in fn_mtr.items():
				print("\t{}, {:.3f}".format(metric, metric_val))
		for k, v in mtr_all.items():
			print('{}: {:.3f}, {:.3f}'.format(k, np.mean(v), np.std(v)))
		curr_hmp_roc = get_roc_info(lst_lbl, lst_scr)
		curr_hmp_pr  = get_pr_info(lst_lbl, lst_scr)
		roc_dict[idx] = curr_hmp_roc
		pr_dict[idx] = curr_hmp_pr
	legend_dict = {0: ('magenta', first_ext), 1: ('green', second_ext)}
	# lstm_ext = lstm_dir_rsl.replace(' ', '_').replace(':', '')
	lstm_ext = lstm_dir_rsl.split(os.sep)[-3]
	fig_name = f'{tiff_dir}/combined_roc_from_{os.path.basename(lstm_ext)}.tiff'
	plot_curves(roc_dict, legend_dict, 'roc', fig_name)
	fig_name = f'{tiff_dir}/combined_pr_from_{os.path.basename(lstm_ext)}.tiff'
	plot_curves(pr_dict, legend_dict, 'pr', fig_name)

if __name__ == '__main__':
	main()
