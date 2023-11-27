from avro.datafile import DataFileReader
from avro.io import DatumReader
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
import json
import os
import sys
import glob 
import pytz
import numpy as np
import pandas as pd
import re
import subprocess

i = int(sys.argv[1])
user_id = sys.argv[2]

#define path, folders, uzaser 
participant_data_path = '/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/aws_data/1/1/participant_data/' # path to the folder that contains folder for each date
output_folder = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/raw_data/{user_id}/eda/' #output folder


# accelerometer data 
dfs = pd.DataFrame()
dates = sorted(os.listdir(participant_data_path), reverse=True) #all date-folders available 
dates.remove('.DS_Store')

if (f'empatica_eda_{user_id}'+dates[i]+'.csv') not in sorted(os.listdir(output_folder),reverse=True):
    folder = os.listdir(participant_data_path+dates[i]) # list folders (for each user) within the date-folde
    subfolder = glob.glob(os.path.join(participant_data_path, f'*{dates[i]}*/*{user_id}*//raw_data/v6/')) #path to avro files (within date->within user)    
    if  subfolder != []:
        if os.path.isdir(subfolder[0]):
            files = os.listdir(subfolder[0]) #list of avro files
            files = np.sort(files).tolist() # rearrange files in a chronological manner
            for ff in files: #loop through files to read and store data
                avro_file = subfolder[0]+ff
                reader = DataFileReader(open(avro_file, "rb"), DatumReader())
                schema = json.loads(reader.meta.get('avro.schema').decode('utf-8'))
                data = []
                for datum in reader:
                    data = datum
                reader.close()

                eda = data['rawData']['eda']
                startSeconds = eda["timestampStart"] / 1000000
                timeSeconds = list(range(0,len(eda['values'])))
                #datetime_timetemp = [datetime.utcfromtimestamp(x) for x in timeUNIXtemp]
                if eda["samplingFrequency"] == 0:
                    eda["samplingFrequency"] = 64;
                timeUNIX = [t/eda["samplingFrequency"]+startSeconds for t in timeSeconds]
                df_eda = pd.concat([pd.DataFrame(timeUNIX), pd.DataFrame(eda['values'])],axis = 1)                
                df_eda.columns = ['time','eda']
                dfs = pd.concat([dfs,df_eda])
                if not df_eda.empty:
                    df_eda.columns = ['time','eda']
                    dfs = pd.concat([dfs,df_eda])
                    
            dfs=dfs.reset_index(drop = True)
            dfs.to_csv(output_folder+f'empatica_eda_{user_id}_'+dates[i]+'.csv')
            dfs = pd.DataFrame()
            print(dates[i])
else:
    print('file already exists')
    
