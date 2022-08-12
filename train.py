"""
Created on Wed Oct 27 11:09:36 2021

@author: cxue2
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import sys
import train_func as tf
from handle_input import get_args
from misc import gen_seed_dirs
from read_txt import select_task

def main():
	"""
	main entrypoint;
	"""
	args = get_args(sys.argv[1:])
	args = {k: v for k, v in args.items() if v is not None}
	task_csv_txt = args.get('task_csv_txt', 'task_csvs.txt')
	task_id = args.get('task_id', 0)
	device = int(args.get('device', 0))
	n_epoch = int(args.get('n_epoch', 1))
	num_seeds = int(args.get('num_seeds', 1))
	num_folds = int(args.get('num_folds', 5))
	do_rand_seg = args.get('do_rand_seg')
	num_pt_segments = int(args.get('num_pt_segments', 10))
	pt_segment_root = args.get('pt_segment_root')
	seg_min = int(args.get('seg_min', 5))
	learning_rate = float(args.get('learning_rate', 1e-3))
	negative_loss_weight = float(args.get('negative_loss_weight', 1))
	positive_loss_weight = float(args.get('positive_loss_weight', 1))
	weights = [negative_loss_weight, positive_loss_weight]
	debug_stop = args.get('debug_stop')
	no_save_model = args.get('no_save_model')
	no_write_fold_txt = args.get('no_write_fold_txt')

	final_args = {'task_id': task_id,
		'device': device, 'n_epoch': n_epoch,
		'num_seeds': num_seeds, 'num_folds': num_folds,
		'do_rand_seg': do_rand_seg, 'num_pt_segments': num_pt_segments,
		'pt_segment_root': pt_segment_root, 'seg_min': seg_min,
		'learning_rate': learning_rate, 'weights': weights,
		'no_save_model': no_save_model}

	csv_info, ext = select_task(task_id, task_csv_txt)

	seed_to_dir = gen_seed_dirs(num_seeds, ext, n_epoch)
	for seed, dir_rsl in seed_to_dir.items():
		time_ext, trn_dir, vld_dir = tf.gen_dirs(dir_rsl, time_ext)
		for vld_idx in range(num_folds):
			for tst_idx in range(num_folds):
				if vld_idx == tst_idx:
					continue ## vld and tst fold can't be the same?
				dset_kw = tf.set_dset_kw(num_folds, vld_idx, tst_idx, seed, do_rand_seg,
					num_pt_segments, pt_segment_root, seg_min)
				dset_trn, dset_vld, dset_tst = tf.gen_audio_datasets(csv_info, dset_kw)

				model_obj = tf.fit_model(device, n_epoch, learning_rate, weights, debug_stop,
					dset_trn, dset_vld, dir_rsl)

				vld_tst = f'vld_{vld_idx}_tst_{tst_idx}'

				tf.save_model(no_save_model, dir_rsl, model_obj, vld_tst)

				x_fp_to_rsl = model_obj.eval(dset_tst, b_size=16)
				df_dat = dset_tst.df_dat

				tf.save_vectors(dir_rsl, vld_tst, df_dat, x_fp_to_rsl)
				tf.save_csvs(df_dat, dset_trn, dset_vld, dir_rsl, seed, vld_tst, trn_dir, vld_dir)
				tf.write_fold_txt(no_write_fold_txt, dir_rsl, vld_tst, ext, seed, vld_idx, tst_idx,
					final_args, dset_trn, dset_vld, dset_tst)

if __name__ == '__main__':
	main()
