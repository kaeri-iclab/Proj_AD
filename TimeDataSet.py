#%%
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os, glob, json, pickle
import pandas as pd
import numpy as np

class TimeseriesAnomalyDataset(torch.utils.data.Dataset): 
    def __init__(self, TARGET_DATA, samprate, window, stride, json_root, scaler='MMSC', timestamp = False, fileformat='pkl', transient=True, ABNORM_TRJ_CLASS=[], ABNORM_STEP_CLASS=[]):
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
        self.TIMESTAMP=timestamp
        self.FFORMAT=fileformat
        self.TRANSIENT=transient
        # Load pretrained sklearn scaler
        if scaler is not None:
            with open(self._ROOT_+'/scaler.pkl','rb') as f: 
                self.SCALER = pickle.load(f)[scaler]
        else:
            self.SCALER = None
        
        # Create file list according to the predetermined file format
        self.file_list=[]
        for i in self.NORM_TRJ_CLASS:
            self.file_list += glob.glob(self._ROOT_+f'/{i}/*.{self.FFORMAT}')

        # Load and manipulate data
        self.X_DATA = []  # To save np.array of sampled trajectory
        X_NORMAL = []  # To create index mapper idx > (TRJ index, STP index)

        for i in self.file_list:
            with open(i,'rb') as f:
                if self.FFORMAT == 'pkl' or self.FFORMAT == 'pickle':
                    temp = pickle.load(f)[::self.SAMPRATE] # read dataframe with samprate interval
                    temp.reset_index(drop=True,inplace=True) # reset index
                elif self.FFORMAT=='npy':
                    temp = np.load(f)[::self.SAMPRATE]        
                else: 
                    raise NotImplementedError
        
            # Drop timestamp column
            if self.TIMESTAMP==True:
                if isinstance(temp,pd.DataFrame):
                    temp.drop([temp.columns[0]], axis=1, inplace=True)
                elif isinstance(temp,np.ndarray):
                    temp=temp[:,1:]   

            temp=np.array(temp)
            # Normalizing variable except the last column (i.e., cls)
            if self.SCALER is not None:
                scaled_temp = self.SCALER.transform(temp[:,:-1])                
                self.X_DATA.append(scaled_temp) 
            else:
                self.X_DATA.append(temp[:,:-1]) 

            # Select row where STEP_CLASS defined as NORMAL
            normal_index = np.where(np.isin(temp[:,-1], self.NORM_STEP_CLASS))[0]

            # Among selected rows, get class information with stride, where the index is greater than WINDOW-1)
            # e.g., if window is 10, we can read 0~9 (len=10), but not -1~8
            base_normal_index = normal_index[normal_index>=(self.WINDOW-1)][::self.STRIDE]
            if self.TRANSIENT: #if transient is true then transient part(different step class in window) can be used               
                X_NORMAL.append(pd.Series(temp[base_normal_index,-1],
                            index = base_normal_index))
            else : # otherwise all time step should have same stepclass in given window.       
                normal_index_wo_transient=[]
                for j in base_normal_index:
                    classes_in_window = temp[j-self.WINDOW+1:j+1,-1]
                    if all(classes_in_window == temp[j,-1]):
                        normal_index_wo_transient.append(j)
                X_NORMAL.append(pd.Series(temp[normal_index_wo_transient,-1],
                            index = np.array(normal_index_wo_transient)))
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
        return torch.Tensor(self.X_DATA[int(tr_idx)][int(step_idx)-self.WINDOW+1:int(step_idx)+1,:]), torch.Tensor([int(step_cls)])
