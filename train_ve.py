"""
Created on Wed Oct 27 11:09:36 2021

@author: cxue2
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import os
import sys
import numpy as np
import fhs_split_dataframe as sdf
from audio_dataset import AudioDataset
from handle_input import get_args
from misc import gen_seed_dirs, get_time
from model import Model
from mri_func import get_target_vectors, gan_get_target_vectors
from net_ve import VoiceEncoder
from read_txt import select_task

def main():
	"""
	main entrypoint;
	"""
	args = get_args(sys.argv[1:])
	args = {k: v for k, v in args.items() if v is not None}
	task_csv_txt = args.get('task_csv_txt', 'task_csvs.txt')
	task_id = args.get('task_id', 0)
	img_pt_txt = args.get('img_pt_txt', 'img_pt.txt')
	img_pt_idx = args.get('img_pt_idx', 0)
	encode_idx = args.get('encode_idx', 0)
	## 0=cnn img autoencoder, 1=gan img autoencoder
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

	final_args = {'task_id': task_id, 'img_pt_idx': img_pt_idx,
		'device': device, 'n_epoch': n_epoch,
		'num_seeds': num_seeds, 'num_folds': num_folds,
		'do_rand_seg': do_rand_seg, 'num_pt_segments': num_pt_segments,
		'pt_segment_root': pt_segment_root, 'seg_min': seg_min,
		'learning_rate': learning_rate, 'weights': weights,
		'no_save_model': no_save_model}
	csv_info, ext = select_task(task_id, task_csv_txt)
	get_encoded = get_target_vectors
	if int(encode_idx) == 1:
		get_encoded = gan_get_target_vectors
		assert img_pt_txt == "gan_pt.txt"
		ext += "_gan"
	mni_fp_to_vector = get_encoded(img_pt_idx, img_pt_txt, csv_info)
	seed_to_dir = gen_seed_dirs(num_seeds, ext, n_epoch)
	for seed, dir_rsl in seed_to_dir.items():
		time_ext = get_time()
		dir_rsl = f'{dir_rsl}/{time_ext}'
		assert not os.path.isdir(dir_rsl), dir_rsl
		os.makedirs(dir_rsl)
		print(dir_rsl)
		trn_dir = f'{dir_rsl}/trn'
		vld_dir = f'{dir_rsl}/vld'
		for dir_to_make in [trn_dir, vld_dir]:
			if not os.path.isdir(dir_to_make):
				os.makedirs(dir_to_make)
		for vld_idx in range(num_folds):
			for tst_idx in range(num_folds):
				if vld_idx == tst_idx:
					continue
					## vld and tst fold can't be the same?
				dset_kw = {'num_folds': num_folds, 'vld_idx': vld_idx, 'tst_idx': tst_idx,
					'seed': seed, 'mni_fp_to_vector': mni_fp_to_vector}
				if do_rand_seg:
					yield_data_and_target_kw = {'num_pt_segments': num_pt_segments,
						'pt_segment_root': pt_segment_root,
						'seg_min': seg_min}
					dset_kw.update({'yield_data_and_target': sdf.yield_rand_seg_and_mni,
						'yield_data_and_target_kw': yield_data_and_target_kw})

				dset_trn = AudioDataset(csv_info, 'TRN', **dset_kw)
				dset_vld = AudioDataset(csv_info, 'VLD', **dset_kw)
				dset_tst = AudioDataset(csv_info, 'TST', **dset_kw)

				n_concat = 10
				model_obj = Model(n_concat, nn=VoiceEncoder(), device=device)
				model_fit_kw = {'n_epoch': n_epoch, 'b_size': 4, 'learning_rate': learning_rate,
					'weights': weights, 'debug_stop': debug_stop}
				model_obj.fit(dset_trn, dset_vld, dir_rsl, **model_fit_kw)
				vld_tst = f'vld_{vld_idx}_tst_{tst_idx}'
				if not no_save_model:
					pt_file_dir = f"{dir_rsl}/pt_files/"
					if not os.path.isdir(pt_file_dir):
						os.makedirs(pt_file_dir)
					model_obj.save_model(os.path.join(pt_file_dir, f'{vld_tst}.pt'))
				x_fp_to_rsl = model_obj.eval(dset_tst, b_size=16)
				df_dat = dset_tst.df_dat
				voice_mri_vector_parent = f'{dir_rsl}/voice_mri_vectors/{vld_tst}'
				if not os.path.isdir(voice_mri_vector_parent):
					os.makedirs(voice_mri_vector_parent)
				for df_idx, row in df_dat.iterrows():
					audio_fn = row['audio_fn']
					start, end = row['start'], row['end']
					voice_mri_vector = x_fp_to_rsl[audio_fn][(start, end)]
					base_audio = os.path.splitext(os.path.basename(audio_fn))[0]
					voice_mri_vector_fp = os.path.join(voice_mri_vector_parent,
						f'voice_mri_{base_audio}_{start}_{end}.npy')
					np.save(voice_mri_vector_fp, voice_mri_vector)
					df_dat.loc[df_idx, 'voice_mri_vector_fp'] = voice_mri_vector_fp
				df_dat.to_csv(f'{dir_rsl}/tst_audio_{seed}_{vld_tst}.csv', index=False)
				dset_trn.df_dat.to_csv(f'{trn_dir}/trn_audio_{seed}_{vld_tst}.csv', index=False)
				dset_vld.df_dat.to_csv(f'{vld_dir}/vld_audio_{seed}_{vld_tst}.csv', index=False)
				if not no_write_fold_txt:
					txt_fp = os.path.join(dir_rsl, f"comb_{vld_tst}.txt")
					with open(txt_fp, 'w') as outfile:
						outfile.write(f'ext={ext}; seed={seed}; ')
						outfile.write(f'vld_idx={vld_idx}; tst_idx={tst_idx};\n')
						outfile.write("".join([f'{k}: {v}; ' for k, v in final_args.items()]) +\
							"\n")
						outfile.write(f"\nTRN IDs: {dset_trn.patient_list}\n\n")
						outfile.write(f"VLD IDs: {dset_vld.patient_list}\n\n")
						outfile.write(f"TST IDs: {dset_tst.patient_list}\n\n")

if __name__ == '__main__':
	main()
