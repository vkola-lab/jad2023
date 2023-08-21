#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_subplots.py
plot four curves into one figure;
"""
import os
import sys
import glob
from collections import defaultdict
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from misc import calc_performance_metrics
from misc import get_roc_info, get_pr_info
from multi_curves import plot_curve

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['savefig.facecolor'] = 'w'

def collect_output(string, string_list, only_print=False):
	"""
	collect output or just print;
	"""
	if only_print:
		print(string)
	else:
		string_list.append(string)

def write_txt(string_list, txt_out):
	"""
	write txt;
	"""
	with open(txt_out, 'a') as outfile:
		for string in string_list:
			outfile.write(f'{string}\n')

def get_dir_list(parent_dir):
	"""
	get list of directories;
	"""
	return [os.path.join(parent_dir, d) for d in os.listdir(parent_dir)]

def plot_subplot_curve(ax, hmp_roc, legend_dict, curve_str):
	"""
	plot all the curves;
	"""
	legend_str = {}

	color, legend_ext = legend_dict[0]
	p_mean, _, auc_mean, auc_std = plot_curve(curve_str, ax, hmp_roc['xs'],
		hmp_roc['ys_mean'], hmp_roc['ys_upper'],
		hmp_roc['ys_lower'], hmp_roc['auc_mean'], hmp_roc['auc_std'],
		color=color)
	msg = r'{}: {:.3f}$\pm${:.3f}'.format(legend_ext, auc_mean, auc_std)
	legend_str[0] = (p_mean, msg)
	p_mean_list = [v[0] for k, v in legend_str.items()]
	msg_list = [v[1] for k, v in legend_str.items()]
	ax.legend(p_mean_list, msg_list,
			  facecolor='w', prop={"weight":'bold', "size":15},
			  bbox_to_anchor=(0.04, 0.04, 0.5, 0.5),
			  loc='lower left')

def plot_individual_curve(hmp_roc, legend_dict, curve_str, fig_name):
	"""
	plot all the curves;
	"""
	fig, ax = plt.subplots(1)

	legend_str = {}

	color, legend_ext = legend_dict[0]
	p_mean, _, auc_mean, auc_std = plot_curve(curve_str, ax, hmp_roc['xs'],
		hmp_roc['ys_mean'], hmp_roc['ys_upper'],
		hmp_roc['ys_lower'], hmp_roc['auc_mean'], hmp_roc['auc_std'],
		color=color)
	msg = r'{}: {:.3f}$\pm${:.3f}'.format(legend_ext, auc_mean, auc_std)
	legend_str[0] = (p_mean, msg)
	p_mean_list = [v[0] for k, v in legend_str.items()]
	msg_list = [v[1] for k, v in legend_str.items()]
	ax.legend(p_mean_list, msg_list,
			  facecolor='w', prop={"weight":'bold', "size":17},
			  bbox_to_anchor=(0.04, 0.04, 0.5, 0.5),
			  loc='lower left')

	fig.savefig(fig_name, dpi=300, format='svg', bbox_inches='tight')
	plt.close('all')

def main():
	"""
	main entrypoint
	results/HC_vs_DE_tscript_osm_and_mfcc_npy_1.0_1.0_lr_nworkers0_orig_tcn/32_epochs/21639
	results/NDE_vs_DE_tscript_osm_and_mfcc_npy_1.0_1.0_lr_nworkers0_orig_tcn/32_epochs/75533
	"""
	dir_list = sys.argv[1:]
	## subplot dimensions should be 2 (roc, pr) X len(dir_list)
	dir_ext = '_'.join([d.split(os.sep)[-1] for d in dir_list])
	now = str(datetime.now()).replace(' ', '_').replace(':', '_')
	parent_dir = f'plot_subplots/{now}/'
	fig_name = os.path.join(parent_dir, f'{dir_ext}.svg')
	if not os.path.isdir(os.path.dirname(fig_name)):
		os.makedirs(os.path.dirname(fig_name))
	fig_size = len(dir_list) * 5
	fig, axs = plt.subplots(len(dir_list), 2, figsize=(fig_size, fig_size))

	roc_to_title = {0: '(A)', 1: '(C)', 2: '(E)'}
	pr_to_title = {0: '(B)', 1: '(D)', 2: '(F)'}
	string_list = []
	only_print = False
	for idx, dir_rsl in enumerate(dir_list):
		current_string_list = [dir_rsl + "\n"]
		mode = 'chunk'
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
		print(dir_rsl)
		print(len(lst_csv))
		if lst_csv == [] or len(lst_csv) not in [5, 20]:
			continue
		print(f"{len(lst_csv)} csvs found;")
		fn_metrics = {}
		for fn in lst_csv:
			df = pd.read_csv(fn)
			# get scores and labels
			if mode == 'chunk':
				lbl = df.label.to_numpy()
				scr = df.score.to_numpy()
			elif mode == 'audio_avg':
				tmp = df.groupby('audio_fn').mean().to_numpy()
				lbl = tmp[:,0].astype(np.int)
				scr = tmp[:,-1]
			else:
				raise AssertionError(f'invalid mode: {mode}')
			mtr = calc_performance_metrics(scr, lbl)
			for k, mtr_val in mtr.items():
				if k == 'mat':
					continue
				mtr_all[k].append(mtr_val)
			fn_metrics[fn] = {mk: mv for mk, mv in mtr.items() if mk != 'mat'}
			lst_lbl.append(lbl)
			lst_scr.append(scr)
		print(f'loaded {len(fn_metrics)} results;')
		for filename, fn_mtr in fn_metrics.items():
			collect_output(filename, current_string_list, only_print=only_print)
			for metric, metric_val in fn_mtr.items():
				collect_output("\t{}, {:.3f}".format(metric, metric_val), current_string_list,
					only_print=only_print)
		collect_output('avg_performance:', current_string_list, only_print=only_print)
		for k, v in mtr_all.items():
			collect_output('{}: {:.3f}, {:.3f}'.format(k, np.mean(v), np.std(v)),
				current_string_list, only_print=only_print)
		try:
			curr_hmp_roc = get_roc_info(lst_lbl, lst_scr)
			curr_hmp_pr  = get_pr_info(lst_lbl, lst_scr)
		except (TypeError, ValueError) as error:
			print(error)
			continue
		legend_dict = {0: ('magenta', 'AUC')}

		plot_subplot_curve(axs[idx, 0], curr_hmp_roc, legend_dict, 'roc')
		axs[idx, 0].set_title(roc_to_title[idx], fontweight='bold')
		plot_individual_curve(curr_hmp_roc, legend_dict, 'roc',
			os.path.join(parent_dir, f'{idx}_roc.svg'))
		plot_subplot_curve(axs[idx, 1], curr_hmp_pr, legend_dict, 'pr')
		plot_individual_curve(curr_hmp_pr, legend_dict, 'pr',
			os.path.join(parent_dir, f'{idx}_pr.svg'))
		axs[idx, 1].set_title(pr_to_title[idx], fontweight='bold')
		string_list.extend(current_string_list)
	fig.savefig(fig_name, dpi=300, format='svg')
	fig.savefig(fig_name.replace('.svg', '.tiff'), dpi=100, format='tiff')
	plt.close('all')
	print(fig_name)

	parent_dir = sys.argv[1]
	parent_dir_ext = os.path.normpath(parent_dir).split(os.sep)[-2]
	now = str(datetime.now()).replace(' ', '_').replace(':', '_')
	txt_out = os.path.join('txt', str(datetime.now()).split(' ', maxsplit=1)[0], parent_dir_ext)
	if not os.path.isdir(txt_out):
		os.makedirs(txt_out)
	txt_out = os.path.join(txt_out, f'{now}_output.txt')
	write_txt(string_list, txt_out)
	print(txt_out)

if __name__ == '__main__':
	main()
