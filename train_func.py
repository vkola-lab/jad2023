"""
train_func.py
train functions
"""
import os
import numpy as np
from misc import get_time
from tcn import TCN

def gen_dirs(dir_rsl):
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
	return trn_dir, vld_dir

def set_dset_kw(num_folds, vld_idx, tst_idx, seed, do_rand_seg, num_pt_segments, pt_segment_root,
	seg_min, audio_idx, get_label):
	"""
	setting dataset kwargs
	"""
	dset_kw = {'num_folds': num_folds, 'vld_idx': vld_idx, 'tst_idx': tst_idx,
		'seed': seed, 'audio_idx': audio_idx, 'get_label': get_label}
	if do_rand_seg:
		yield_data_and_target_kw = {'num_pt_segments': num_pt_segments,
			'pt_segment_root': pt_segment_root,
			'seg_min': seg_min}
		dset_kw.update({'yield_data_and_target': None,
			'yield_data_and_target_kw': yield_data_and_target_kw})
	return dset_kw

def gen_audio_datasets(csv_info, dset_kw, audio_dset):
	"""
	get audio datasets
	"""
	dset_trn = audio_dset(csv_info, 'TRN', **dset_kw)
	dset_vld = audio_dset(csv_info, 'VLD', **dset_kw)
	dset_tst = audio_dset(csv_info, 'TST', **dset_kw)
	return dset_trn, dset_vld, dset_tst

def get_model(model, ys_len, channels, device, feat_indices):
	"""
	get model
	"""
	tcn_kw = {'ys_len': ys_len, 'channels': channels, 'feat_indices': feat_indices}
	neural = TCN(device, **tcn_kw)
	model_obj = model(10, neural, device=device)
	return model_obj

def fit_model(n_epoch, learning_rate, weights, debug_stop, dset_trn, dset_vld, dir_rsl,
	loss_fn, model_obj):
	"""
	fit model
	"""
	model_fit_kw = {'n_epoch': n_epoch, 'b_size': 4, 'learning_rate': learning_rate,
		'weights': weights, 'debug_stop': debug_stop, 'loss_fn': loss_fn}
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
		mfcc_fp = row['mfcc_fp']
		start, end = row['start'], row['end']
		voice_fsl_vector = x_fp_to_rsl[mfcc_fp][(start, end)]
		base_audio = os.path.splitext(os.path.basename(mfcc_fp))[0]
		fname = f'voice_fsl_{base_audio}_{start}_{end}.npy'
		fname = fname.replace('_None_None', '')
		voice_fsl_vector_fp = os.path.join(voice_fsl_vector_parent, fname)
		np.save(voice_fsl_vector_fp, voice_fsl_vector)
		df_dat.loc[df_idx, 'voice_fsl_vector_fp'] = voice_fsl_vector_fp

def save_csvs(df_dat, dset_trn, dset_vld, dir_rsl, seed, vld_tst, trn_dir, vld_dir):
	"""
	saving tst, trn, vld csvs;
	"""
	df_dat.to_csv(f'{dir_rsl}/tst_audio_{seed}_{vld_tst}.csv', index=False)
	dset_trn.df_dat.to_csv(f'{trn_dir}/trn_audio_{seed}_{vld_tst}.csv', index=False)
	dset_vld.df_dat.to_csv(f'{vld_dir}/vld_audio_{seed}_{vld_tst}.csv', index=False)

def dset_info_to_txt(dataset, ext):
	"""
	write dset info to txt;
	"""
	line = f"{ext}: num_patients: {dataset.num_patients}, num_audio: "+\
		f"{dataset.num_audio} [negative_audio={dataset.num_negative_audio}, "+\
		f"positive_audio={dataset.num_positive_audio}]\n"
	return line

def write_fold_txt(no_write_fold_txt, dir_rsl, vld_tst, ext, seed, vld_idx, tst_idx, final_args,
	dset_trn, dset_vld, dset_tst):
	"""
	write fold txt;
	"""
	if not no_write_fold_txt:
		txt_fp = os.path.join(dir_rsl, f"comb_{vld_tst}.txt")
		trn_line = dset_info_to_txt(dset_trn, 'TRN')
		vld_line = dset_info_to_txt(dset_vld, 'VLD')
		tst_line = dset_info_to_txt(dset_tst, 'TST')
		with open(txt_fp, 'w') as outfile:
			outfile.write(f'ext={ext}; seed={seed}; ')
			outfile.write(f'vld_idx={vld_idx}; tst_idx={tst_idx};\n')
			outfile.write("".join([f'{k}: {v}; ' for k, v in final_args.items()]) +\
				"\n")
			outfile.write(trn_line)
			outfile.write(vld_line)
			outfile.write(tst_line)
			outfile.write(f"\nTRN IDs: {dset_trn.patient_list}\n\n")
			outfile.write(f"VLD IDs: {dset_vld.patient_list}\n\n")
			outfile.write(f"TST IDs: {dset_tst.patient_list}\n\n")
