#import necessary modules
from dicom2nifti import dicom_series_to_nifti
import numpy as np
import dicom2nifti
import matplotlib.pyplot as plt
import os
import pandas as pd

df=pd.read_csv('repeat_dicom_to_nifti.csv')

for i,row in df.iterrows():
	try:
		dicom_series_to_nifti(row['neuroml_mri_dir'],row['to_t1_nifti'])
	except:
		print('Found error with {}'.format(row['neuroml_mri_dir']))

