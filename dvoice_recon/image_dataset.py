import warnings
warnings.filterwarnings("ignore")

import torch
import cupy as cp
import numpy as np
import pandas as pd
from os.path import exists
import nibabel as nib
from cupyx.scipy.ndimage import zoom
import tqdm

class ImageDataset(torch.utils.data.Dataset): 
    
    def __init__(self, csv, selections, mode, split, seed=3227, dim_z=None, preload=False):

        self._dim_z = dim_z
        
        # read raw csv
        df = pd.read_csv(csv)
        
        # filter dataframe by selections
        msk = self._get_mask_selections(df, selections)
        df = df[msk]
        df.reset_index(drop=True, inplace=True)
        
        # filter dataframe by MRI files' existence
        msk = self._get_mask_mri_existence(df)
        df = df[msk]
        df.reset_index(drop=True, inplace=True)
        
        # filter dataframe by modes
        msk = self._get_mask_mode(df, mode, split, seed)
        df = df[msk]
        df.reset_index(drop=True, inplace=True)
        
        # done
        self.df = df

        # preload
        self._preload = preload
        self.mri = np.empty((len(self.df), 1, 64, 64, 64), dtype=np.float32)
        if preload:
            for idx in tqdm.tqdm(range(len(self.df))):
                fn = '{}/{}'.format(self.df.at[idx, 'path'], self.df.at[idx, 'filename'])
                img = cp.array(nib.load(fn).dataobj) if fn.endswith('.nii') else cp.load(fn)

                # zoom
                img = img[:, 18:-18, :]
                img = zoom(img, 0.352)

                # to numpy
                img = cp.asnumpy(img)
                self.mri[idx, 0, :, :, :] = img
        
    def __len__(self):
        
        return len(self.df)    
    
    def __getitem__(self, idx):

        if self._preload:
            img = self.mri[idx]
        
        else:
            # load mri
            fnm = '{}/{}'.format(self.df.at[idx, 'path'], self.df.at[idx, 'filename'])
            img = cp.array(nib.load(fnm).dataobj) if fnm.endswith('.nii') else cp.load(fnm)

            # zoom
            img = img[:, 18:-18, :]
            img = zoom(img, 0.352)

            # add channel dimension
            img = img[np.newaxis, :]

            # to numpy
            img = cp.asnumpy(img)

        # for autoencoder
        if self._dim_z is None:
            return img, img, idx

        # for adversarial autoencoder
        else:
            z = torch.randn((self._dim_z,), dtype=torch.float32)
            return img, z, idx
    
    def _get_mask_selections(self, df, selections):
        
        msk = np.ones((len(df), len(selections)), dtype=bool)
        
        for i, (k, v) in enumerate(selections.items()):
            
            if type(k) is tuple:
                
                tmp = [tuple(l) for l in df[list(k)].values.tolist()]
                msk[:, i] = [t in v for t in tmp]
            
            else:    
            
                msk[:, i] = df[k].isin(v).tolist()
        
        return np.all(msk, axis=1).tolist()             
    
    def _get_mask_mri_existence(self, df):
        
        msk = []
        
        for i in range(len(df)):   
            fnm = '{}/{}'.format(df.at[i, 'path'], df.at[i, 'filename'])
            
            if exists(fnm):
                msk += [True]
                
            else:             
                msk += [False]
                print('{} does not exist.'.format(fnm))
                
        return msk
    
    def _get_mask_mode(self, df, mode, split, seed):
        
        # normalize split into ratio
        ratio = np.array(split) / np.sum(split)
        
        # list of modes for all samples
        arr = []
        
        # 0th ~ (N-1)th modes
        for i in range(len(split) - 1):       
            arr += [i] * round(ratio[i] * len(df))
        
        # last mode
        arr += [len(split) - 1] * (len(df) - len(arr))
        
        # random shuffle
        arr = np.array(arr)
        np.random.seed(seed)
        np.random.shuffle(arr)
        
        # generate mask
        msk = (arr == mode).tolist()
        
        return msk


if __name__ == '__main__':
    
    selections = {'dataset': ['ADNI1', 'ADNI2', 'ADNI3'],
                  #'dataset': ['NACC_ALL'],
                  ('NC', 'AD'): [(0, 1), (1, 0)]}   
    
    dst_trn = ImageDataset('../data/mri_ayan.csv', selections, 0, [4, 1])
    
    print(dst_trn.df)

    print(dst_trn[0][0].shape)
    # print(dst_trn[0][0])
