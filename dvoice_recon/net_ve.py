#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 15:40:27 2021

@author: cxue2
"""
import torch
import torch.nn as nn

class VoiceEncoder(nn.Module):
	"""
	VoiceEncoder model
	"""
	def __init__(self):

		super(VoiceEncoder, self).__init__()
		# CNN
		self.module = nn.Sequential(

			# nn.BatchNorm1d(13),
			nn.Conv1d(in_channels=13, out_channels=32, kernel_size=1, stride=1, padding=0),

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

			# global average pooling
			nn.AdaptiveAvgPool1d(1),

			# squeeze
			nn.Flatten(),)

	def forward(self, xs):
		"""
		pass forward;
		"""
		out = []
		for x in xs:
			tmp = self.module(x)
			out.append(tmp)
		out = torch.cat(out)
		return out

	def get_scores_loss(self, Xs, ys, loss_fn):
		"""
		get scores and loss;
		"""
		scores = self(Xs)
		loss = loss_fn(scores, ys)
		return scores, loss

	def get_scores(self, Xs):
		"""
		get scores;
		"""
		return self(Xs)

	def reformat(self, Xs, _):
		"""
		reformat to tensor;
		"""
		for idx, _ in enumerate(Xs):
			Xs[idx] = torch.tensor(Xs[idx], dtype=torch.float32,
				device=self.device)
			Xs[idx] = Xs[idx].permute(1, 0)
			Xs[idx] = Xs[idx].view(1, Xs[idx].shape[0], Xs[idx].shape[1])

if __name__ == '__main__':
	x = [torch.rand(1, 13, 160000),
		 torch.rand(1, 13, 40000)]
	m = VoiceEncoder()
	o = m(x)
	print(o.shape)
