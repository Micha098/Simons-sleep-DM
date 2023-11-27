import heartpy as hp
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
from datetime import timedelta
from datetime import time
import neurokit2 as nk
import json
import sys
import os
import glob 
import pytz
import numpy as np
import pandas as pd
import re
import scipy
from scipy import signal
from scipy.signal import find_peaks,butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import mode
from avro.datafile import DataFileReader
from avro.io import DatumReader
import subprocess

def z_score_outlier_filter(data, threshold):
    mean = np.mean(data)
    std = np.std(data)
    z_scores = [(x - mean) / std for x in data]
    filtered_data = [x if np.abs(z_scores[i]) <= threshold else (data[i-1] + data[i+1])/2 for i, x in enumerate(data[:-1])]
    return filtered_data



i = int(sys.argv[1]) -1 
date = sys.argv[2]

subject_id = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['id'].tolist()

#define path, folders, uzaser 
participant_data_path = '/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/aws_data/1/1/participant_data/' # path to the folder that contains folder for each date
output_folder = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/raw_data/{subject_id[i]}/bvp/' #output folder
execute_preprocessing = True

# bvp data 

dfs = pd.DataFrame()
dates = sorted(os.listdir(participant_data_path), reverse=True) #all date-folders available 
dates.remove('.DS_Store')

filename = f'empatica_bvp_{subject_id[i]}_'+date+'.csv'

if (filename in sorted(os.listdir(output_folder),reverse=True)):
    if (os.path.getsize(output_folder+filename) > 300000000):
        print(f'{filename} already exist')
        execute_preprocessing = False

elif execute_preprocessing:

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
            print('finished preprocessing '+date)
    else:
        print(f'subfolfer {date} empty')



# i = int(sys.argv[1])-1
target_date = sys.argv[2]


input_folder = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/raw_data/{subject_id[i]}/bvp/' #output folder
output_folder =  f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/measure_data/{subject_id[i]}/hr/' 

files = sorted(os.listdir(input_folder))

#files.remove('.DS_Store')

path_bvp = None  # Initialize path_edf outside the loop

for pathi in files:
    try:
        file_date = re.search(r'\d{4}-\d{2}-\d{2}', pathi).group()
        print(file_date)
    except:
        continue
    if file_date:        
        if file_date == target_date:
            path_bvp = pathi
            file_date = dt.datetime.strptime(file_date, '%Y-%m-%d').date()
            break

if path_bvp:
    
    FileName = f'{input_folder}{path_bvp}'
    New_fileName = f'hr{path_bvp[12:]}'

    if (New_fileName in sorted(os.listdir(output_folder),reverse=True)):
        if (os.path.getsize(output_folder+New_fileName) > 90000):
            print(f'{New_fileName} already exist')
            sys.exit()

    else:        
        
        if (f'{subject_id[i]}' in path_bvp) & (FileName.endswith('.csv')):

            data = pd.read_csv(FileName,index_col=0)
            #try:
            data['date'] = data['timestamp'].apply(lambda x: dt.datetime.utcfromtimestamp(x))
            data.datetime_time = pd.to_datetime(data['datetime_time'])

            # fill gaps in raw date with ffil method 
            data = data.sort_values(by='datetime_time')

            date_range = pd.date_range(start=data['datetime_time'].min(), end=data['datetime_time'].max(), freq='15.625L')
            data = data.set_index('datetime_time').reindex(date_range, method='ffill')

            df, info = nk.ppg_process(data["bvp"], sampling_rate=64)


            boundary = 0.5
            fs = 64

            ppg_range = df.PPG_Clean.max() + abs(df.PPG_Clean.min())
            ppg_data = df.PPG_Clean

            #ppg_data = signals.PPG_Clean

            # Remove motion artifacts with band-stop filter
            f0 = 0.5 # Hz
            f1 = 4 # Hz

            b, a = signal.butter(4, [f0, f1], "bandpass", fs=fs)
            ppg_data = signal.filtfilt(b, a, ppg_data)


            epoch_length = int(fs * 60) # 60 seconds
            overlap_length = int(fs * 30) # 30 seconds
            epoch_start = 0
            epoch_end = epoch_start + epoch_length

            initial_hr_data = []
            smoothed_hr = []

            while epoch_end - epoch_length <= len(ppg_data):

                epoch_data = ppg_data[epoch_start:epoch_end]

                peaks, _ = signal.find_peaks(epoch_data, distance=fs*0.2
                                            ,prominence = ppg_range*0.001)


                if len(peaks) > 0:
                    rr_interval = np.diff(peaks) / fs # time between peaks in sec
                    heart_rate = 60 / rr_interval # average heart rate in beats per minute

                    for i in range(len(heart_rate)):
                        if heart_rate[i] < 40 or heart_rate[i] > 180:
                            if i > 0 and i < len(heart_rate)-1:
                                heart_rate[i] = (heart_rate[i-1] + heart_rate[i+1]) / 2

                    heart_rate = np.clip(heart_rate, 40, 180)  # Clip HR values to the range [40, 180]
                    heart_rate_minute = np.median(heart_rate)
                else: 
                    heart_rate_minute = np.nan

                smoothed_hr.append(heart_rate_minute)

                if len(smoothed_hr) % 2 == 0:
                    avg_hr = np.mean(smoothed_hr)
                    initial_hr_data.append(avg_hr)
                    smoothed_hr = []

                epoch_start += overlap_length
                epoch_end += overlap_length


            print(len(initial_hr_data))
            initial_hr_data = z_score_outlier_filter(initial_hr_data, threshold=2)

            result_hr = pd.DataFrame(data.set_index('date').bvp.resample('1T').mean())
            result_hr['HR'] = np.nan
            result_hr['HR'].iloc[:len(initial_hr_data)] = initial_hr_data
            #result_hr['HR'] = result_hr['HR'].interpolate(method='linear')


            result_hr.to_csv(f'{output_folder}/{New_fileName}' )

            print(f'finished processing {New_fileName}')