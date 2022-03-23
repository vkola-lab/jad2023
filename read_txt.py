"""
read_txt.py
read txt files;
"""

def select_task(task_id, task_csv_txt):
	"""
	based on task_id, and return csv_info, ext;
	"0=norm vs. demented;\n1=nondemented vs. demented;\n2=norm vs. mci;\n"+\
				"3=mci vs. demented;"
	"""
	task_id = int(task_id)
	csv_info = None
	with open(task_csv_txt, 'r') as infile:
		for idx, line in enumerate(infile):
			if idx == task_id:
				csv_info, ext = line.split(',')
				ext = ext.strip()
				return csv_info, ext
	return None, None

def get_pt_file(pt_idx, pt_txt):
	"""
	read pt txt file and return the appropriate pretrained(pt) model file based on pt_idx;
	"""
	with open(pt_txt, 'r') as infile:
		for idx, line in enumerate(infile):
			if idx == pt_idx:
				return line
	return None

def get_tst_csv(tst_idx, tst_txt):
	"""
	get tst csv;
	"""
	return get_pt_file(tst_idx, tst_txt)
