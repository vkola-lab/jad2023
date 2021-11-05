"""
calc_metrics.py
calculate metrics
"""
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score
from scipy import interp

def calc_performance_metrics(scr, lbl):
	"""
	calculate performance metrics;
	"""
	met = {}
	# prediction
	prd = (scr > .5) * 1
	# metrics
	met['mat'] = confusion_matrix(y_true=lbl, y_pred=prd)
	TN, FP, FN, TP = met['mat'].ravel()
	N = TN + TP + FN + FP
	S = (TP + FN) / N
	P = (TP + FP) / N
	sen = TP / (TP + FN)
	spc = TN / (TN + FP)
	met['acc'] = (TN + TP) / N
	met['balanced_acc'] = (sen + spc) / 2
	met['sen'] = sen
	met['spc'] = spc
	met['prc'] = TP / (TP + FP)
	met['f1s'] = 2 * (met['prc'] * met['sen']) / (met['prc'] + met['sen'])
	met['wt_f1s'] = f1_score(lbl, prd, average='weighted')
	met['mcc'] = (TP / N - S * P) / np.sqrt(P * S * (1-S) * (1-P))
	try:
		met['auc'] = roc_auc_score(y_true=lbl, y_score=scr)
	except KeyboardInterrupt as kbi:
		raise kbi
	except:
		met['auc'] = np.nan
	return met

def show_performance_metrics(met):
	"""
	print performance metrics;
	"""
	met_list = ['mat', 'acc', 'sens', 'spc', 'prc', 'f1s', 'mcc', 'auc']
	for item in met_list:
		if item == 'mat':
			to_prt = np.array_repr(met[item]).replace("\n", "")
			print(f'\t{item}: {to_prt}')
		print(f'\t{item}: {met[item]}')
