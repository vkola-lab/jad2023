"""
train_func.py
train functions
"""
import os
import numpy as np
import fhs_split_dataframe as sdf
from audio_dataset import AudioDataset
from misc import get_time
from model import Model
from tcn import TCN

def gen_dirs(dir_rsl, time_ext):
	"""
	generate result dir, trn dir, vld dir
	"""
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
	return time_ext, trn_dir, vld_dir

def set_dset_kw(num_folds, vld_idx, tst_idx, seed, do_rand_seg, num_pt_segments, pt_segment_root,
	seg_min):
	"""
	setting dataset kwargs
	"""
	dset_kw = {'num_folds': num_folds, 'vld_idx': vld_idx, 'tst_idx': tst_idx,
		'seed': seed}
	if do_rand_seg:
		yield_data_and_target_kw = {'num_pt_segments': num_pt_segments,
			'pt_segment_root': pt_segment_root,
			'seg_min': seg_min}
		dset_kw.update({'yield_data_and_target': sdf.yield_rand_seg_and_mni,
			'yield_data_and_target_kw': yield_data_and_target_kw})
	return dset_kw

def gen_audio_datasets(csv_info, dset_kw):
	"""
	get audio datasets
	"""
	dset_trn = AudioDataset(csv_info, 'TRN', **dset_kw)
	dset_vld = AudioDataset(csv_info, 'VLD', **dset_kw)
	dset_tst = AudioDataset(csv_info, 'TST', **dset_kw)
	return dset_trn, dset_vld, dset_tst

def fit_model(device, n_epoch, learning_rate, weights, debug_stop, dset_trn, dset_vld, dir_rsl):
	"""
	fit model
	"""
	model_obj = Model(10, nn=TCN(device), device=device)
	model_fit_kw = {'n_epoch': n_epoch, 'b_size': 4, 'learning_rate': learning_rate,
		'weights': weights, 'debug_stop': debug_stop}
	model_obj.fit(dset_trn, dset_vld, dir_rsl, **model_fit_kw)
	return model_obj

def save_model(no_save_model, dir_rsl, model_obj, vld_tst):
	"""
	save model
	"""
	if not no_save_model:
		pt_file_dir = f"{dir_rsl}/pt_files/"
		if not os.path.isdir(pt_file_dir):
			os.makedirs(pt_file_dir)
		model_obj.save_model(os.path.join(pt_file_dir, f'{vld_tst}.pt'))

def save_vectors(dir_rsl, vld_tst, df_dat, x_fp_to_rsl):
	"""
	saving resulting vectors
	"""
	voice_fsl_vector_parent = f'{dir_rsl}/voice_fsl_vectors/{vld_tst}'
	if not os.path.isdir(voice_fsl_vector_parent):
		os.makedirs(voice_fsl_vector_parent)
	for df_idx, row in df_dat.iterrows():
		audio_fn = row['audio_fn']
		start, end = row['start'], row['end']
		voice_fsl_vector = x_fp_to_rsl[audio_fn][(start, end)]
		base_audio = os.path.splitext(os.path.basename(audio_fn))[0]
		voice_fsl_vector_fp = os.path.join(voice_fsl_vector_parent,
			f'voice_fsl_{base_audio}_{start}_{end}.npy')
		np.save(voice_fsl_vector_fp, voice_fsl_vector)
		df_dat.loc[df_idx, 'voice_fsl_vector_fp'] = voice_fsl_vector_fp

def save_csvs(df_dat, dset_trn, dset_vld, dir_rsl, seed, vld_tst, trn_dir, vld_dir):
	"""
	saving tst, trn, vld csvs;
	"""
	df_dat.to_csv(f'{dir_rsl}/tst_audio_{seed}_{vld_tst}.csv', index=False)
	dset_trn.df_dat.to_csv(f'{trn_dir}/trn_audio_{seed}_{vld_tst}.csv', index=False)
	dset_vld.df_dat.to_csv(f'{vld_dir}/vld_audio_{seed}_{vld_tst}.csv', index=False)

def write_fold_txt(no_write_fold_txt, dir_rsl, vld_tst, ext, seed, vld_idx, tst_idx, final_args,
	dset_trn, dset_vld, dset_tst):
	"""
	write fold txt;
	"""
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
