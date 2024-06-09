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

i = int(sys.argv[1]) -1 
date = sys.argv[2]

subject_id = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['id'].tolist()

#define path, folders, uzaser 
participant_data_path = '/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/aws_data/1/1/participant_data/' # path to the folder that contains folder for each date
output_folder = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/raw_data/{subject_id[i]}/acc/' #output folder


# accelerometer data 
dfs = pd.DataFrame()
dates = sorted(os.listdir(participant_data_path), reverse=True) #all date-folders available 
dates.remove('.DS_Store')

if True:#(f'empatica_acc_{subject_id[i]}'+date+'.csv') not in sorted(os.listdir(output_folder),reverse=True):
    folder = os.listdir(participant_data_path+date) # list folders (for each user) within the date-folde
    subfolder = glob.glob(os.path.join(participant_data_path, f'*{date}*/*{subject_id[i]}*//raw_data/v6/')) #path to avro files (within date->within user)    
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
                    acc = datum["rawData"]["accelerometer"]  # access specific metric
                    startSeconds = acc["timestampStart"] / 1000000  # convert timestamp to seconds
                    timeSeconds = list(range(0, len(acc['x'])))
                    if acc["samplingFrequency"] == 0:
                        acc["samplingFrequency"] = 64
                    timeUNIX = [t / acc["samplingFrequency"] + startSeconds for t in timeSeconds]
                    delta_physical = acc["imuParams"]["physicalMax"] - acc["imuParams"]["physicalMin"]
                    delta_digital = acc["imuParams"]["digitalMax"] - acc["imuParams"]["digitalMin"]
                    acc['x'] = [val * delta_physical / delta_digital for val in acc["x"]]
                    acc['y'] = [val * delta_physical / delta_digital for val in acc["y"]]
                    acc['z'] = [val * delta_physical / delta_digital for val in acc["z"]]

                    df_acTot = pd.DataFrame({'time': timeUNIX, 'x': acc['x'], 'y': acc['y'], 'z': acc['z']})

                    if not df_acTot.empty:
                        dfs.append(df_acTot)

            if dfs:
                daily_df = pd.concat(dfs, ignore_index=True)
                daily_filename = f'{tz_temp}/empatica_acc_{subject_id[i]}_{date}.csv'
                daily_df.to_csv(daily_filename)

                print(f'Data for {date} saved.')
                dfs = []
else:
    print('acc file already exists')


#define path, folders, uzaser 
participant_data_path = '/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/aws_data/1/1/participant_data/' # path to the folder that contains folder for each date
output_folder = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/raw_data/{subject_id[i]}/bvp/' #output folder

# bvp data 

dfs = pd.DataFrame()
dates = sorted(os.listdir(participant_data_path), reverse=True) #all date-folders available 
dates.remove('.DS_Store')
if (f'empatica_bvp_{subject_id[i]}'+date+'.csv') not in sorted(os.listdir(output_folder),reverse=True):

    folder = os.listdir(participant_data_path+date) # list folders (for each user) within the date-folde
    subfolder = glob.glob(os.path.join(participant_data_path, f'*{date}*/*{subject_id[i]}*//raw_data/v6/')) #path to avro files (within date->within user)    
    if  subfolder != []:
        if os.path.isdir(subfolder[0]):
            files = os.listdir(subfolder[0]) #list of avro files
            files = np.sort(files).tolist()# rearrange files in a chronological manner
            for ff in files: #loop through files to read and store dataֿ
                avro_file = subfolder[0]+ff
                reader = DataFileReader(open(avro_file, "rb"), DatumReader())
                schema = json.loads(reader.meta.get('avro.schema').decode('utf-8'))
                data = []
                for datum in reader:
                    data = datum
                reader.close()

                bvp = data["rawData"]["bvp"] #access specific metric 
                if len(data["rawData"]["bvp"]['values']) > 0:

                    startSeconds = bvp["timestampStart"] / 1000000 # convert timestamp to seconds
                    timeSeconds = list(range(0,len(bvp['values'])))
                    timeUNIX = [t/bvp["samplingFrequency"]+startSeconds for t in timeSeconds]
                    datetime_time = [datetime.utcfromtimestamp(x) for x in timeUNIX]

                    df_bvpTot = pd.concat([pd.DataFrame(timeUNIX), pd.DataFrame(datetime_time),pd.DataFrame(bvp['values'])],axis = 1)
                    df_bvpTot.columns = ['timestamp','datetime_time','bvp']
                    dfs = pd.concat([dfs,df_bvpTot])
            dfs=dfs.reset_index(drop = True)
            dfs.to_csv(output_folder+f'empatica_bvp_{subject_id[i]}_'+date+'.csv')
            dfs = pd.DataFrame()
            print(f'{date}')
else:
    print('ppg file already exists')

#define path, folders, uzaser 
participant_data_path = '/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/aws_data/1/1/participant_data/' # path to the folder that contains folder for each date
output_folder = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/raw_data/{subject_id[i]}/eda/' #output folder


# accelerometer data 
dfs = pd.DataFrame()
dates = sorted(os.listdir(participant_data_path), reverse=True) #all date-folders available 
dates.remove('.DS_Store')

if (f'empatica_eda_{subject_id[i]}'+date+'.csv') not in sorted(os.listdir(output_folder),reverse=True):
    folder = os.listdir(participant_data_path+date) # list folders (for each user) within the date-folde
    subfolder = glob.glob(os.path.join(participant_data_path, f'*{date}*/*{subject_id[i]}*//raw_data/v6/')) #path to avro files (within date->within user)    
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
            dfs.to_csv(output_folder+f'empatica_eda_{subject_id[i]}_'+date+'.csv')
            dfs = pd.DataFrame()
            print(date)
else:
    print('eda file already exists')
    