from torch.utils.data import WeightedRandomSampler
import torch
from imbalanced_mri_loader import  MRIDataset,map_disease_status,get_loader_from_spreadsheet
import pandas as pd
import torch.optim as optim
from convolutional_backbones import ConvLayer, vanilla_CNN_backbone
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split m
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from autoencoder_utils import encoder_decoder,pad_case,save_checkpoint, evaluate_AE_epoch
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs=200
learning_rate=0.0003

#load in data, then split into testing and training sets
to_csv='/home/mattmill/Desktop/aibl_adni_combined.csv'
df=pd.read_csv(to_csv)
train_df,test_df=train_test_split(df,test_size=0.2,random_state=42,shuffle=True)

#Define dataset and dataloader classes
train_dataset=MRIDataset(train_df)
test_dataset=MRIDataset(test_df)

#define hyperparameters
criterion=nn.MSELoss()
batch_size=8

#get your loaders
train_loader=get_loader_from_spreadsheet(train_df)
test_loader=get_loader_from_spreadsheet(test_df)

#prep tensorboard for tracking
writer=SummaryWriter(f'auto_encoder_runs/autoencoder_with_internal_MLP/first_try_32filters')

#initialize autoencoder
autoencoder=encoder_decoder(1,0.4).to(device)

#initialize gradient descent optimizer
optimizer=optim.Adam(autoencoder.parameters(),lr=learning_rate,weight_decay=1e-5)

step=0 


#


for epoch in range(num_epochs):
        
    running_loss=0
    print(f"Now training epoch {epoch} of {num_epochs}") #with hidden dim {hidden_dim}, dropout {dropout}, and batch size {batch_size}")
    for i,(images,labels) in enumerate(tqdm((train_loader))):
    #for i,(images,labels) in train_loader:
        #print("Now training epoch_{}, batch_{}".format(epoch,i))
        
        #zero your gradients
        optimizer.zero_grad()
        
        
        images=torch.unsqueeze(images,1)
        images=pad_case(images) #ensure that the images are of correct shape for entry into network
        #put your images and labels on the device
        images=images.to(device)
        
        print('Now calculating the loss')
        #with torch.cuda.amp.autocast():
        #run your model
        outputs=autoencoder(images)
        

        #update the model
        loss=criterion(outputs,images) #in the case of SSIM, you create loss by 1-criterion()
        running_loss+=loss.item()
        
        print('Loss calculated!')
        
        loss.backward()
        optimizer.step()

        #optimizer.zero_grad()
        #calculates your gradients
        #scaler.scale(loss).backward()
        
        #updates parameters
        #scaler.step(optimizer)
        #scaler.update()
        

        
    
    epoch_train_loss=running_loss/len(train_dataset)
    epoch_test_loss=evaluate_AE_epoch(test_loader,autoencoder,criterion)
    print(f"Epoch {epoch},  Training Loss {epoch_train_loss}, Testing Loss {epoch_test_loss}")
    
    #record epoch-wise loss and accuracy, append it to list of epochs in training
    #epoch_train_accuracy,epoch_train_loss,train_ground_truths,epoch_train_scores=evaluate_epoch(train_loader,full_pipeline,criterion)
    #epoch_test_accuracy,epoch_test_loss,test_ground_truths,epoch_test_scores=evaluate_epoch(test_loader,full_pipeline,criterion)
        
    #epoch_loss=running_loss/len(train_dataset)
    #epoch_train_accuracy=check_accuracy(train_loader,full_pipeline)
    #epoch_test_accuracy = check_accuracy(test_loader,full_pipeline)
    
    #print(f"Epoch {epoch}, Train Loss {epoch_train_loss}, Train Accuracy: {epoch_train_accuracy}, Test Loss: {epoch_test_loss},Test Accuracy: {epoch_test_accuracy}")
    
    #add to tensorboard
    writer.add_scalar("Training Loss",epoch_train_loss,global_step=step)
    writer.add_scalar("Testing Loss",epoch_test_loss,global_step=step)
#    writer.add_scalar("Training Accuracy",epoch_train_accuracy,global_step=step)
#    writer.add_scalar("Testing Loss",epoch_test_loss,global_step=step)
#    writer.add_scalar("Testing Accuracy",epoch_test_accuracy,global_step=step)
    step+=1 #this will make the writer move on
    
    if (epoch+1)%50==0:
        print('Now saving another checkpoint!')
        checkpoint = {"state_dict": autoencoder.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint,filename='autoencoder_With_MLP_checkpoint_epoch_{}.pth'.format(epoch+1))
    

#    if epoch_loss < best_test_accuracy:
#        print("Found new best testing accuracy!")
#        best_test_accuracy=epoch_test_accuracy
#        checkpoint = {"state_dict": autoencoder.state_dict(), "optimizer": optimizer.state_dict()}
#        save_checkpoint(checkpoint,filename='mri_checkpoint.pth')
    
    #save preds and scores for plotting
#        best_ground_truths=np.array(test_ground_truths)
#        np.save("ground_truths.npy",best_ground_truths)
#        best_prediction_scores=np.array(epoch_test_scores)
#        np.save("best_prediction_scores.npy",best_prediction_scores)






