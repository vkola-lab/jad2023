# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 16:52:17 2020

@author: Iluva
"""
import torch
import torch.nn as nn

class TCN(nn.Module):
	"""
	TCN class - 2022-10-31
	"""
	def __init__(self, device, **kwargs):
		"""
		constructor
		"""
		super(TCN, self).__init__()
		self.ys_len = kwargs.get('ys_len')
		channels = kwargs.get('channels')
		self.device = device     # 'cpu' or 'cuda:x'
		self.tcn = nn.Sequential(
			# nn.BatchNorm1d(13),
			nn.Conv1d(in_channels=channels, out_channels=32, kernel_size=1, stride=1, padding=0),

			# nn.BatchNorm1d(32),
			nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
			nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
			nn.ELU(),
			nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
			# nn.Dropout(),

			# nn.BatchNorm1d(32),
			nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
			nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
			nn.ELU(),
			nn.MaxPool1d(kernel_size=4, stride=4, padding=0),
			# nn.Dropout(),

			# nn.BatchNorm1d(64),
			nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
			nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
			nn.ELU(),
			nn.MaxPool1d(kernel_size=4, stride=4, padding=0),

			# nn.BatchNorm1d(128),
			nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
			nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
			nn.ELU(),
			nn.MaxPool1d(kernel_size=4, stride=4, padding=0),

			# nn.BatchNorm1d(128),
			nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.ELU(),
			nn.MaxPool1d(kernel_size=4, stride=4, padding=0),

			# nn.BatchNorm1d(256),
			nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.ELU(),
			nn.MaxPool1d(kernel_size=4, stride=4, padding=0),

			# nn.BatchNorm1d(256),
			nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ELU(),
			nn.MaxPool1d(kernel_size=4, stride=4, padding=0),

			# nn.BatchNorm1d(512),
			nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.ELU(),
		)
		# linear layer
		self.mlp = nn.Sequential(
			nn.Linear(512, self.ys_len, bias=False)
		)

		self.tcn.to(device)
		self.mlp.to(device)

	def forward(self, Xs):
		"""
		fwd
		"""
		out = []
		for X in Xs:
			tmp = self.tcn(X)
			# global average pooling
			tmp = torch.mean(tmp, dim=2)
			# linear layer
			tmp = self.mlp(tmp)
			tmp = tmp.squeeze()
			out.append(tmp)
		out = torch.stack(out)
		return out

	def forward_wo_gpool(self, Xs):
		"""
		fwd without gpool
		"""
		out = []
		for X in Xs:
			tmp = self.tcn(X)
			out.append(tmp)
		return out

	def get_scores(self, Xs):
		"""
		get scores;
		"""
		return self(Xs)

	def get_scores_loss(self, Xs, ys, loss_fn, target=None):
		"""
		get scores and loss;
		"""
		scores = self(Xs)
		if len(scores.shape) == 1:
			scores = torch.unsqueeze(scores, 1)
		# Xs = np.array(Xs)
		if target is None:
			loss = loss_fn(scores, ys)
		else:
			loss = loss_fn(scores, ys, target)
		return scores, loss

	def reformat(self, Xs, _):
		"""
		reformat Xs array accordingly;
		"""
		for idx, _ in enumerate(Xs):
			Xs[idx] = torch.tensor(Xs[idx], dtype=torch.float32,
				device=self.device)
			Xs[idx] = Xs[idx].permute(1, 0)
			Xs[idx] = Xs[idx].view(1, Xs[idx].shape[0], Xs[idx].shape[1])

if __name__ == '__main__':
	i = [torch.rand(1, 13, 160000).to(0),
		 torch.rand(1, 13, 40000).to(0)]
	m = TCN(0, channels=13, ys_len=2)
	o = m(i)
	print(o)
	print(o.shape)
