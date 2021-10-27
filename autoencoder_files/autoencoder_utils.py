from torch.utils.data import WeightedRandomSampler
import torch
from imbalanced_mri_loader import  MRIDataset,map_disease_status,get_loader_from_spreadsheet
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.nn.functional import softmax
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard

    
class encoder_decoder(nn.Module):
    "Encoder that encodes Scan to vector"
    def __init__(self, in_size, drop_rate, fil_num=32,
                 out_channels=1):
        super().__init__()
        self.in_size = in_size
        # MaxPool3d(kernel_size, stride=None, padding=0,...)
        self.conv1 = nn.Conv3d(in_size, fil_num, 5, 3, 2, bias=False)
        self.bn1 = nn.BatchNorm3d(fil_num)
        self.conv2 = nn.Conv3d(fil_num, 4*fil_num, 5, 3, 2, bias=False)
        self.bn2 = nn.BatchNorm3d(4*fil_num)
        self.conv3 = nn.Conv3d(4*fil_num, 16*fil_num, 5, 3, 2, bias=False)
        self.bn3 = nn.BatchNorm3d(16*fil_num)
        self.conva = nn.LeakyReLU()
        self.dr = nn.Dropout(drop_rate)
        
        
        
        
        self.deconv1=nn.ConvTranspose3d(16*fil_num,4*fil_num,5,3,1,output_padding=0)
        self.bn4=nn.BatchNorm3d(4*fil_num)
        self.deconv2=nn.ConvTranspose3d(4*fil_num,fil_num,5,3,1,output_padding=0)
        self.bn5=nn.BatchNorm3d(fil_num)
        self.deconv3=nn.ConvTranspose3d(fil_num,self.in_size,5,3,1,output_padding=0)
        self.bn6=nn.BatchNorm3d(self.in_size)
        self.reverse_linear=nn.Linear(512,512*7*8*7) #this is the layer that will build up o
        kernel_size=(7,8,7) #hard coded at the moment 
        self.avg_pool=nn.AvgPool3d(kernel_size)
        
        
        self.encoder=nn.Sequential(
                self.conv1,
                self.bn1,
                self.conva,
                self.conv2,
                self.bn2,
                self.conva,
                self.conv3,
                self.bn3,
                self.conva,
                self.avg_pool)
        
        self.decoder=nn.Sequential(
                self.deconv1,
                self.bn4,
                self.conva,
                self.deconv2,
                self.bn5,
                self.conva,
                self.deconv3,
                self.bn6,
                self.conva)


    def forward(self, x):
        #encoder block
        x=self.encoder(x)
      
        #now we will take out the latent vector: this is ultimately what will be used for training the speech encoder
        x=x.view(-1,512)
        
        #now get latent vector back up to a size that is suitable for the decoder portion
        x=self.reverse_linear(x)
                
        #now get latent representation back up to a SHAPE that is suitable for decoder portion
        x=x.view(-1,512,7,8,7)
        
        #decoder block
        x=self.decoder(x)
        
        return x
    
    
def pad_case(tensor):
    tensor=tensor[:,:,0:181,0:216,0:181]
    return F.pad(tensor,(4,4,0,0,4,4),"constant",0)


def evaluate_AE_epoch(loader,model,criterion):
    """Check MSE Loss on validation set after each epoch"""

    running_loss=0
    model.eval()
    with torch.no_grad():
        for data, targets in loader:
            data=torch.unsqueeze(data,1).to(device)
            data=pad_case(data)
            #targets=targets.to(device)
            
            #run data through the model
            outputs=model(data)

            
            #loss
            loss=criterion(outputs,data)
            running_loss+=loss.item()
    model.train()
    

    overall_loss=running_loss/len(test_dataset)
    
    return overall_loss


def save_checkpoint(state, filename="mri_checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    