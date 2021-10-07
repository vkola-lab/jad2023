import subprocess
import os
import glob
import timeit

#start timer
start = timeit.default_timer()

#define directory paths; will be called within shell script as positional args
root = '/encryptedfs/data/mri_data/voice_pts/vkola_2021_06_17/t1_mpr_niftis'
to_mris = root + '/mri'
to_tmp = root + '/tmp'
to_seg = root + '/seg'
to_MNI_brains = root + '/MNI_brains'


#######
for file in glob.glob(to_mris + '/*nii.gz'):
	study=file.split('/')[-1] #just get the base file name
	study_file_only_name = study.split('.')[0] #eg, just 'sub-20_T1w'
	print(study_file_only_name)
	try:
		subprocess.call('bash ./master_registration_pipeline.sh '+to_mris + ' ' + study+ ' ' + to_tmp + ' ' + to_MNI_brains + ' ' +to_seg + ' '+ study_file_only_name,shell=True)
	except:
		print('Something bad happened on {}'.format(study_file_only_name))

#stop timer and calculate runtime:
stop = timeit.default_timer()
print('Total runtime: ', stop - start)
