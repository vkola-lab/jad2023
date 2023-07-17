"""
model.py
overarching model class that can take in any neural network object;
"""
import sys
from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm
from binary_audio_dataset import collate_fn
from misc import calc_performance_metrics, show_performance_metrics

class BinaryModel:
	"""
	overarching model class;
	"""

	def __init__(self, n_concat, nn=None, device='cpu'):
		"""
		init;
		"""
		self.n_concat = n_concat
		self.nn = nn
		self.set_nn_device(device)
		self.device = device

	def set_nn_device(self, device):
		"""
		call model.to() and set device;
		check range of gpu devices (currently have 4 gpus);
		"""
		assert device in ['cpu', 0, 1, 2, 3], 'Invalid device.'
		self.nn.to(device)
		self.nn.device = device

	def fit(self, dset_trn, dset_vld, dir_rsl, **kwargs):
		"""
		fit method
		"""
		n_epoch = kwargs.get('n_epoch', 32)
		b_size = kwargs.get('b_size', 4)
		learning_rate = kwargs.get('learning_rate', 0.001)
		weights = kwargs.get('weights', [])
		debug_stop = kwargs.get('debug_stop', False)
		loss_fn = kwargs.get('loss_fn', torch.nn.CrossEntropyLoss)
		shuffle = True ## changed 09-20-22
		## changing num_workers to 0 on 2022-10-31;
		dataloader_kw = {'batch_size': b_size, 'shuffle': shuffle, 'num_workers': 0,
			'collate_fn': collate_fn, 'sampler': None}
		dldr_trn = torch.utils.data.DataLoader(dset_trn, **dataloader_kw)
		weights = torch.FloatTensor(weights).cuda(self.device)
		opt = torch.optim.Adam(self.nn.parameters(), lr=learning_rate)
		loss_fn = loss_fn(weight=weights)
		train_epochs_kw = {'debug_stop': debug_stop, 'n_epoch': n_epoch, 'opt': opt,
			'loss_fn': loss_fn, 'dir_rsl': dir_rsl}
		self.train_epochs(dset_trn, dset_vld, dldr_trn, **train_epochs_kw)

	def train_epochs(self, dset_trn, dset_vld, dldr_trn, **kwargs):
		"""
		go thru epochs and train;
		"""
		debug_stop = kwargs.get('debug_stop')
		n_epoch = kwargs.get('n_epoch')
		opt = kwargs.get('opt')
		loss_fn = kwargs.get('loss_fn')
		dir_rsl = kwargs.get('dir_rsl')
		vld_mcc = -1
		if not debug_stop:
			for epoch in range(n_epoch):
				## set model to training mode
				self.nn.train()
				cum_loss, cum_corr, count = 0, 0, 0
				## training loop
				with tqdm(total=len(dset_trn), desc=f'Epoch {epoch} (TRN)',
					ascii=True, bar_format='{l_bar}{r_bar}', file=sys.stdout,
						miniters=len(dset_trn) / 100) as pbar:
					for Xs, ys, audio, _, *additional_feats in dldr_trn:
						## from collate_fn(batch) in binary_audio_dataset.py
						## fea, label, audio_fp, start, end
						## Xs -> list (len=num samples in batch)
						## ys -> numpy.ndarray (len=num samples in batch)
						self.nn.reformat(Xs, self.n_concat)
						ys = torch.tensor(ys, dtype=torch.long, device=self.nn.device)
						self.nn.zero_grad()
						scores, loss = self.nn.get_scores_loss(Xs, ys, loss_fn,
							additional_feats=additional_feats)
						loss.backward()
						opt.step()
						pred = torch.argmax(scores, 1)
						## accumulated loss
						cum_loss += loss.data.cpu().numpy() * len(ys)
						## accumulated no. of correct predictions
						cum_corr += (pred == ys).sum().data.cpu().numpy()
						## accumulated no. of processed samples
						count += len(ys)
						## update statistics and progress bar
						pbar.set_postfix({
							'loss': f'{(cum_loss / count):.6f}',
							'acc' : f'{(cum_corr / count):.6f}',
						})
						pbar.update(len(ys))
				## forward validation dataset
				scr = self.prob(dset_vld)
				met = calc_performance_metrics(scr, dset_vld.labels)
				print('Audio-level validation performance:')
				show_performance_metrics(met)
				print()
				# save model
				if np.isnan(met['mcc']):
					continue
				if vld_mcc <= met['mcc']:
					vld_mcc = met['mcc']
					self.save_model(f'{dir_rsl}/tmp.pt')
			# load best model
			if dset_vld is not None and vld_mcc != -1:
				self.load_model(f'{dir_rsl}/tmp.pt')

	def prob(self, dset, b_size=16, eval_collate_fn=collate_fn):
		"""
		calc model output
		"""
		_, rsl = self.eval(dset, b_size=b_size, eval_collate_fn=eval_collate_fn)
		return np.exp(rsl)[:,1] / np.sum(np.exp(rsl), axis=1)

	def eval(self, dset, **kwargs):
		"""
		evaluate model;
		"""
		b_size = kwargs.get('b_size', 16)
		debug_stop = kwargs.get('debug_stop', False)
		eval_collate_fn = kwargs.get('eval_collate_fn', collate_fn)
		if debug_stop:
			return []
		self.nn.eval()
		## changing num_workers to 0 on 2022-11-01;
		dl_dr_kwargs = {'batch_size': b_size,
				  'shuffle': False,
				  'num_workers': 0,
				  'collate_fn': eval_collate_fn}
		dldr = torch.utils.data.DataLoader(dset, **dl_dr_kwargs)
		# list to store result (i.e. all outputs)
		x_fp_to_rsl = defaultdict(dict)
		all_results = []
		# evaluation loop
		with torch.set_grad_enabled(False):
			with tqdm(total=len(dset), desc='Epoch ___ (EVL)', ascii=True,
				bar_format='{l_bar}{r_bar}', file=sys.stdout, miniters=len(dset) / 100) as pbar:
				for Xs, _, x_filepaths, start_end_list, *additional_feats in dldr:
					self.nn.reformat(Xs, self.n_concat)
					out = self.nn.forward(Xs, additional_feats)
					# out = self.nn.get_scores(Xs)
					# append batch outputs to result
					results = out.data.cpu().numpy()
					all_results.append(results)
					for idx, x_fp in enumerate(x_filepaths):
						if len(x_fp) == 2:
							x_fp = tuple(x_fp)
						start_end = start_end_list[idx]
						assert start_end not in x_fp_to_rsl[x_fp], f'{x_fp}, {start_end}'
						x_fp_to_rsl[x_fp][start_end] = results[idx]
					# progress bar
					pbar.update(len(Xs))
		all_results = np.concatenate(all_results)
		# concatenate all batch outputs
		return x_fp_to_rsl, all_results

	def save_model(self, filepath):
		"""
		save model as a file;
		"""
		torch.save(self.nn, filepath)

	def load_model(self, filepath):
		"""
		load a model file;
		"""
		self.nn = torch.load(filepath)
