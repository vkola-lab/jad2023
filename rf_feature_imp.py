"""
rf_feature_imp.py
examining random forest feature importance
"""
import os
from datetime import datetime
import pandas as pd
from csv_read import yield_csv_data

def hc_de_func_demo():
	"""
	return hc de func + demographics CSV
	"""
	return "random_forest/2023-08-03/2023-08-03_173351.129095_HC_vs_DE_age_encoded"+\
		"_sex_edu_functionals/feat_imp_age_encoded_sex_edu.csv"

def mci_de_func_demo():
	"""
	MCI DE func + demographics
	"""
	return "random_forest/2023-08-03/2023-08-03_173404.735108_MCI_vs_DE_age_encoded"+\
		"_sex_edu_functionals/feat_imp_age_encoded_sex_edu.csv"

def nde_de_func_demo():
	"""
	NDE DE func + demo
	"""
	return "random_forest/2023-08-03/2023-08-03_173416.467012_NDE_vs_DE_age_encoded"+\
		"_sex_edu_functionals/feat_imp_age_encoded_sex_edu.csv"

def hc_de_func_age():
	"""
	hc de func + age
	"""
	return "random_forest/2023-07-27/2023-07-27_125744.177476/HC_vs_DE_age_functionals/"+\
		"feat_imp_age.csv"

def mci_de_func_age():
	"""
	mci de func + age
	"""
	return "random_forest/2023-07-27/2023-07-27_125758.102821/MCI_vs_DE_age_functionals/"+\
		"feat_imp_age.csv"

def nde_de_func_age():
	"""
	nde de func + age
	"""
	return "random_forest/2023-07-27/2023-07-27_125832.564769/"+\
		"NDE_vs_DE_age_functionals/feat_imp_age.csv"


def hc_de_func():
	"""
	hc de func
	"""
	return "random_forest/2023-07-27/2023-07-27_125818.415604/"+\
		"HC_vs_DE__functionals/feat_imp_.csv"

def mci_de_func():
	"""
	mci de func
	"""
	return "random_forest/2023-07-27/2023-07-27_125831.625829/"+\
		"MCI_vs_DE__functionals/feat_imp_.csv"

def nde_de_func():
	"""
	nde de func
	"""
	return "random_forest/2023-07-27/2023-07-27_125904.853281/"+\
		"NDE_vs_DE__functionals/feat_imp_.csv"

def read_sort(csv_in):
	"""
	read and sort in descending order;
	"""
	to_sort = []
	for row in yield_csv_data(csv_in):
		feat = row['feature_name']
		imp = float(row['importance'])
		to_sort.append((imp, feat))
	return sorted(to_sort, reverse=True)

def gen_csv(hc_de_csv, mci_de_csv, nde_de_csv, csv_ext, cutoff):
	"""
	generate feat imp CSV
	"""
	hc_de = read_sort(hc_de_csv)[:cutoff]
	mci_de = read_sort(mci_de_csv)[:cutoff]
	nde_de = read_sort(nde_de_csv)[:cutoff]
	ext_and_data = {'HC_DE': (hc_de, hc_de_csv),
		'MCI_DE': (mci_de, mci_de_csv),
		'NDE_DE': (nde_de, nde_de_csv)}
	now = str(datetime.now()).replace(' ', '_').replace(':', '_')
	parent_dir = f'rf_feature_imp/{csv_ext}_{now}'
	if not os.path.isdir(parent_dir):
		os.makedirs(parent_dir)
	df_dict = {'ext': [], 'feat': [], 'imp': [], 'csv': []}
	for ext, data in ext_and_data.items():
		data, csv_in = data
		for imp, feat in data:
			df_dict['imp'].append(imp)
			df_dict['feat'].append(feat)
			df_dict['csv'].append(csv_in)
			df_dict['ext'].append(ext)
	pd.DataFrame(df_dict).to_csv(f'{parent_dir}/{csv_ext}.csv', index=False)
	print(parent_dir)

def func(cutoff=10):
	"""
	func analysis
	"""
	gen_csv(hc_de_func(), mci_de_func(), nde_de_func(), 'func_', cutoff)

def func_age(cutoff=10):
	"""
	func age analysis
	"""
	gen_csv(hc_de_func_age(), mci_de_func_age(), nde_de_func_age(), 'func_age', cutoff)

def func_demo(cutoff=10):
	"""
	func demo analysis
	"""
	gen_csv(hc_de_func_demo(), mci_de_func_demo(), nde_de_func_demo(), 'func_demo', cutoff)

def main():
	""""
	main
	"""
	func(cutoff=10)
	# func_age(cutoff=10)
	#func_demo(cutoff=10)

if __name__ == '__main__':
	main()
