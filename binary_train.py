"""
Created on Wed Oct 27 11:09:36 2021

@author: cxue2
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import sys
import torch
import train_func as tf
from binary_audio_dataset import BinaryAudioDataset
from binary_model import BinaryModel
from handle_input import get_args
from load_all_data import load_all_data, load_all_osm_and_mfcc
from misc import gen_seed_dirs, pregen_seed_dirs
from read_txt import select_task

def main():
	"""
	main entrypoint;
	"""
	args = get_args(sys.argv[1:])
	args = {k: v for k, v in args.items() if v is not None}
	task_csv_txt = args.get('task_csv_txt', 'task_csvs.txt')
	task_id = args.get('task_id', 0)
	arg_ext = args.get('arg_ext', '')
	audio_idx = int(args.get('audio_idx', 0))
	device = int(args.get('device', 0))
	n_epoch = int(args.get('n_epoch', 1))
	num_seeds = int(args.get('num_seeds', 1))
	seed_list = args.get('seed_list', [])
	num_folds = int(args.get('num_folds', 5))
	holdout_test = args.get('holdout_test')
	do_rand_seg = args.get('do_rand_seg')
	num_pt_segments = int(args.get('num_pt_segments', 10))
	pt_segment_root = args.get('pt_segment_root')
	seg_min = int(args.get('seg_min', 5))
	learning_rate = float(args.get('learning_rate', 1e-4))
	negative_loss_weight = float(args.get('negative_loss_weight', 1))
	positive_loss_weight = float(args.get('positive_loss_weight', 1))
	weights = [negative_loss_weight, positive_loss_weight]
	debug_stop = args.get('debug_stop')
	no_save_model = args.get('no_save_model')
	no_write_fold_txt = args.get('no_write_fold_txt')

	add_feats = int(args.get('add_feats', 0))
	feats_to_indices = {0: [], 1: ['age'], 2: ['age', 'encoded_sex'],
		3: ['encoded_sex'], 4: ['age', 'encoded_sex', 'edu']}
	feat_indices = feats_to_indices.get(add_feats)
	if add_feats != 0:
		assert feat_indices is not None, f'{add_feats}, {feat_indices}'
	final_args = {'task_id': task_id,
		'device': device, 'n_epoch': n_epoch,
		'num_seeds': num_seeds, 'num_folds': num_folds,
		'do_rand_seg': do_rand_seg, 'num_pt_segments': num_pt_segments,
		'pt_segment_root': pt_segment_root, 'seg_min': seg_min,
		'learning_rate': learning_rate, 'weights': weights,
		'no_save_model': no_save_model, 'add_feats': add_feats,
		'feat_indices': feat_indices}

	csv_info, ext = select_task(task_id, task_csv_txt)

	if audio_idx in [0, 1]:
		channels = 5 if audio_idx == 0 else 13
		audio_idx = 'osm_npy' if audio_idx == 0 else 'mfcc_npy'
		ext += f'_{audio_idx}'
	elif audio_idx == 2:
		channels = 18
		audio_idx = 'osm_and_mfcc_npy'
		ext += f'_{audio_idx}'

	if holdout_test:
		ext += '_holdout_test'

	pos_de_list = [2, 3, 4, 5, 7, 8, 9, 10, 6, 11, 12, 13, 14, 15]
	pos_de_list.extend([16, 17, 18, 19, 20, 21])
	## ge85, ge90 models;
	task_id = int(task_id)
	if task_id in [0, 1]:
		get_label = lambda d: int(d['is_de_and_ad'])
	elif task_id in pos_de_list:
		get_label = lambda d: int(d['is_demented'])
	else:
		raise AssertionError(f'no get label for task id {task_id}')
	audio_dset = BinaryAudioDataset
	loss_fn = torch.nn.CrossEntropyLoss
	ext += f'_{str(negative_loss_weight)}_{str(positive_loss_weight)}'

	ext += f'{arg_ext}'
	if seed_list != []:
		seed_to_dir = pregen_seed_dirs(seed_list, ext, n_epoch)
	else:
		seed_to_dir = gen_seed_dirs(num_seeds, ext, n_epoch)

	if audio_idx == 'osm_and_mfcc_npy':
		# all_npy = preload_mfcc_load_osm(csv_info)
		# get_fea = get_mfcc_osm
		all_npy = load_all_osm_and_mfcc(csv_info)
		get_fea = lambda fp, **kw: kw['all_npy'][fp]
	else:
		all_npy = load_all_data(csv_info, audio_idx)
		get_fea = lambda fp, **kw: kw['all_npy'][fp]
	visited_outer_folds = []
	for seed, dir_rsl in seed_to_dir.items():
		trn_dir, vld_dir = tf.gen_dirs(dir_rsl)
		for vld_idx in range(num_folds):
			for tst_idx in range(num_folds):
				tst_idx = None if holdout_test else tst_idx
				if tst_idx is None and vld_idx in visited_outer_folds:
					continue
				if vld_idx == tst_idx:
					continue ## vld and tst fold can't be the same?
				visited_outer_folds.append(vld_idx)
				dset_kw = tf.set_dset_kw(num_folds, vld_idx, tst_idx, seed, do_rand_seg,
					num_pt_segments, pt_segment_root, seg_min, audio_idx, get_label)
				dset_kw.update({'get_fea': get_fea, 'get_fea_kw': {'all_npy': all_npy},
					'feat_indices': feat_indices})
				dset_trn, dset_vld, dset_tst = tf.gen_audio_datasets(csv_info, dset_kw, audio_dset)
				ys_len = 2 ## - tied to number of output classes?
				model_obj = tf.get_model(BinaryModel, ys_len, channels, device, feat_indices)
				model_obj = tf.fit_model(n_epoch, learning_rate, weights, debug_stop,
					dset_trn, dset_vld, dir_rsl, loss_fn, model_obj)

				vld_tst = f'vld_{vld_idx}_tst_{tst_idx}'

				tf.save_model(no_save_model, dir_rsl, model_obj, vld_tst)

				results = model_obj.prob(dset_tst, b_size=16)
				df_dat = dset_tst.df_dat
				df_dat['score'] = results
				tf.save_csvs(df_dat, dset_trn, dset_vld, dir_rsl, seed, vld_tst, trn_dir, vld_dir)
				tf.write_fold_txt(no_write_fold_txt, dir_rsl, vld_tst, ext, seed, vld_idx, tst_idx,
					final_args, dset_trn, dset_vld, dset_tst)

if __name__ == '__main__':
	main()
