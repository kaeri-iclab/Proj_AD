#%%
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os, glob, json, pickle
import pandas as pd
import numpy as np

class TimeseriesAnomalyDataset(torch.utils.data.Dataset): 
    def __init__(self, TARGET_DATA, samprate, window, stride, json_root, scaler='MMSC', ABNORM_TRJ_CLASS=[], ABNORM_STEP_CLASS=[]):
        self.TARGET_DATA=TARGET_DATA
        with open(json_root,'rb') as f:
            DataInfo=json.load(f)
        self._ROOT_ = DataInfo['ROOT'] + TARGET_DATA
        self.DIM = DataInfo[TARGET_DATA]['TRJ_CLASS']
        self.TRJ_CLASS = DataInfo[TARGET_DATA]['TRJ_CLASS']
        self.STEP_CLASS = DataInfo[TARGET_DATA]['STEP_CLASS']
        self.NORM_TRJ_CLASS = [item for item in self.TRJ_CLASS if item not in ABNORM_TRJ_CLASS]
        self.ABNORM_TRJ_CLASS = ABNORM_TRJ_CLASS
        self.NORM_STEP_CLASS = [item for item in self.STEP_CLASS if item not in ABNORM_STEP_CLASS]
        self.ABNORM_STEP_CLASS = ABNORM_STEP_CLASS
        self.WINDOW = window
        self.SAMPRATE = samprate
        self.STRIDE = stride

        # Load pretrained sklearn scaler
        if scaler is not None:
            with open(self._ROOT_+'/scaler.pkl','rb') as f: 
                self.SCALER = pickle.load(f)[scaler]
        else:
            self.SCALER = None
        
        # Create file list 
        self.file_list=[]
        for i in self.NORM_TRJ_CLASS:
            self.file_list += glob.glob(self._ROOT_+f'/{i}/*.pkl')

        # Load and manipulate data
        # TODO : Add Numpy version
        self.X_DATA = []  # To save np.array of sampled trajectory
        X_NORMAL = []  # To create index mapper idx > (TRJ index, STP index)

        for i in self.file_list:
            with open(i,'rb') as f:
                temp=pickle.load(f)[::self.SAMPRATE] # read dataframe with samprate interval
            temp.reset_index(drop=True,inplace=True) # reset index
            
            # Normalizing variable except the last column (i.e., cls)
            if self.SCALER is not None:
                scaled_temp = self.SCALER.transform(np.array(temp)[:,:-1])                
            else:
                scaled_temp = np.array(temp)[:,:-1]                
            self.X_DATA.append(scaled_temp) 

            # Select row where STEP_CLASS defined as NORMAL
            normal_temp=temp[temp.iloc[:,-1].isin(self.NORM_STEP_CLASS)]
         
            # Among selected rows, get class information with stride, where the index is greater than WINDOW-1)
            # e.g., if window is 10, we can read 0~9 (len=10), but not -1~8
            X_NORMAL.append(normal_temp[normal_temp.index>=(self.WINDOW-1)][::self.STRIDE].iloc[:,-1]) 
        
        # Convert list to dataframe where level 0 is TRJ index and level 1 is STEP index    
        self.NORM_STEP_CLASS_INFO = pd.concat(X_NORMAL,keys=range(len(X_NORMAL))).reset_index()    

    def __len__(self):
        return(len(self.NORM_STEP_CLASS_INFO))
    
    def __getitem__(self, idx): 
        # idx is given by data loader
        # get information from NROM_STEP_CLASS_INFO
        tr_idx, step_idx, step_cls = self.NORM_STEP_CLASS_INFO.loc[idx] 

        # Create windowed data from self.X_DATA and return    
        # from tr_idx-th time series data, if step_idx=10, window=5, get value 6~10, i.e., 6:11
        return torch.Tensor(self.X_DATA[tr_idx][step_idx-self.WINDOW+1:step_idx+1,:]), torch.Tensor([step_cls])
