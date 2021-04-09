#%%
import glob, os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

#%%
# This file is base code for preprocessing the raw data into unified format data
# There can be two type of class : step-wise class, trajectory class
# step class is the label of specific time step t 
# trajectory class is the label of a trajectory. 
# (e.g., time series data obtained by 1 experiment until finish)

#%% 
# Save each trajectory in to _ROOT_/_TRAJECTORY_CLASS_/*.csv or pickle or etc
# Define root
TARGET_DATA = 'MillToolWear'
_ROOT_ = '/RAID8T/Datasets/AD/Processed/{}/'.format(TARGET_DATA)

#%%
# Dataset root : /RAID8T/Datasets/AD/HAR
_BASE_ROOT_ =  '/RAID8T/Datasets/AD/{}/'.format(TARGET_DATA)
file_list = sorted(glob.glob(_BASE_ROOT_+'experiment*'))
TRJ_info = pd.read_csv(_BASE_ROOT_+'train.csv')
pds =[]
for i in range(len(TRJ_info)):
    temp = pd.read_csv(file_list[i])
    mf, pvi = TRJ_info.iloc[i][['machining_finalized','passed_visual_inspection']]
    if pvi =='no':
        temp['pvi']=1
    else :
        temp['pvi']=0
    if mf =='no':
        temp['mf']=1
    else:
        temp['mf']=0
    pds.append(temp)
pds_pd = pd.concat(pds)
file_list_gyro = sorted(glob.glob(_BASE_ROOT_+'gyro*'))
file_list_acc = sorted(glob.glob(_BASE_ROOT_+'acc*'))
df_label = pd.read_csv(_BASE_ROOT_+'train.csv')

#%%
# trajectory class is uid
# step class = 1~12, and 0 for no label
# mmsc : minmaxscaler
# stsc : standardscaler
mmsc = MinMaxScaler()
stsc = StandardScaler()
SCALERS = {'MMSC':MinMaxScaler(), 'STSC':StandardScaler()}

for file_root in tqdm.tqdm(file_list_gyro):
    file_name = os.path.basename(file_root)
    file_name_acc = file_name.replace('gyro','acc')

    # Get experiment id, and user id
    _, eid, uid =file_name.replace('exp','').replace('user','').replace('.txt','').split('_')
    eid = int(eid)
    uid = int(uid)
    # Read to txt files
    df_temp_gyro = pd.read_csv(_BASE_ROOT_+ file_name, header=None, sep=' ', names=['gx','gy','gz'])
    df_temp_acc = pd.read_csv(_BASE_ROOT_ + file_name_acc, header=None, sep=' ', names=['ax','ay','az'])
    df_temp = pd.concat([df_temp_gyro,df_temp_acc],axis=1)
    # Fill NAN value   
    df_temp.fillna(method='pad',axis='columns',inplace=True)
    # df_temp = df_temp.interpolate(axis=1,inplace=True)

    # Train scaler for **input**
    SCALERS['MMSC'].partial_fit(df_temp)
    SCALERS['STSC'].partial_fit(df_temp)

    #Create column for step class
    df_temp['cls']=np.int64(np.zeros(len(df_temp)))
    target_label = df_label[(df_label.eid==eid)&(df_label.uid==uid)]
    
    # Fill step class according to uid and eid
    for idx, temp in target_label.iterrows():
        df_temp['cls'][temp.start:temp.end]=int(temp.label)

    # Save data to pickle
    PATH_TRAJECTORY = _ROOT_+ str(uid)+'/'
    ensure_dir(PATH_TRAJECTORY)
    with open(PATH_TRAJECTORY+str(eid)+'.pkl','wb') as f:
        pickle.dump(df_temp,f)

with open(_ROOT_+'scaler.pkl','wb') as f:
    pickle.dump(SCALERS,f)

# %%
