from torch.utils.data import WeightedRandomSampler
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp



def map_disease_status(string):
    if string=='AD':
        return 1
    else:
        return 0
    
class MRIDataset(Dataset):
    def __init__(self,df,transform=None):
        self.df=df
        self.base_dir='/data/datasets'
        self.transform=transform
        
    def __getitem__(self,index):
        base_name = self.df.iloc[index,1]
        full_scan_path=os.path.join(self.base_dir, base_name)+'.npy'
        
        #now load the scan
        scan = np.load(full_scan_path)
       # scan=full_scan_path
        #now derive the label
        label=map_disease_status(self.df.iloc[index,2])
        
        return scan,label
    
    def __len__(self):
        return len(self.df)

def get_loader_from_spreadsheet(df,batch_size=8):
    
    dataset=MRIDataset(df)
    
    #derive a weighted dataloader from a spreadsheet of cases and AD statuses
    nl_cog_weights=1/len(df[df['status']=='NL'])
    ad_cog_weights=1/len(df[df['status']=='AD'])
    
    class_weights=[nl_cog_weights,ad_cog_weights] #weights for combined ADNI and AIBL --> have to readjust for all cases
    sample_weights=[0]*len(df)
    
    for idx,ad_status in enumerate(df['status']):
        class_weight=class_weights[map_disease_status(ad_status)]
        sample_weights[idx]=class_weight
    
    sampler=WeightedRandomSampler(sample_weights,num_samples=len(sample_weights),replacement=True)
    
    #derive a DataLoader from the MRIDataset and the sampler
    loader=DataLoader(dataset,batch_size=batch_size,sampler=sampler)
    
    return loader
    
