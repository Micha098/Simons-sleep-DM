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
from scipy import signal
import matplotlib.dates as mdates
import seaborn as sns
import datetime
from sklearn import preprocessing
import cProfile

def sadeh_algorithm(epoch_data, window_size=11):

    # Initialize the sleep/awake classification list
    sleep_awake = []

    # Iterate through the epoch data
    for i in range(len(epoch_data)):
        # Get the window of data
        window_start = max(0, i - window_size//2)
        window_end = min(len(epoch_data), i + window_size//2 + 1)
        window = epoch_data[window_start:window_end]

        # Replace missing values with Nones
        window = [None if x is None else x for x in window]

        # Reduce counts over 300 to 300
        window = [min(x, 300) for x in window]

        # Calculate AVG, NATS, SD, LG
        avg = sum(window) / len(window)
        nats = sum(1 for x in window if 50 <= x < 100)
        sd = np.std(window[6:])
        lg = np.log(sum(window[window_size//2-1:window_size//2+2]) + 1)

        # Apply the Sadeh algorithm
        result = 7.601 - (0.065 * avg) - (1.08 * nats) - (0.056 * sd) - (0.703 * lg)

        # Determine sleep/awake classification
        if i < 6:
            sleep_awake.append(0)
        else:
            if result > 0:
                sleep_awake.append(1)
            elif pd.isna(result):
                sleep_awake.append(None)
            else:
                sleep_awake.append(0)

    return sleep_awake

j = int(sys.argv[1]) 

subject_id = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['id'].tolist()
tzs_str = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['tz_str'].tolist()

tz_str = tzs_str[j]

input_folder = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/{subject_id[j]}/acc/' 
output_folder = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/{subject_id[j]}/zcy/' 

if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

files = sorted(os.listdir(input_folder), reverse=True)
date_list = [re.search(r'\d{4}-\d{2}-\d{2}', file).group() for file in os.listdir(input_folder) if file.endswith(".csv") and file.startswith("empatica_acc") ]

date_list = sorted(date_list, reverse=True)

for datei, filei in zip(date_list, files):
    
    file_date = dt.datetime.strptime(datei, '%Y-%m-%d').date()
    print(f'found file: subject_{subject_id[j]} date {file_date}')
    path_acc = filei

    FileName = f'{input_folder}{path_acc}'
    New_fileName = f'empatica_zcy{path_acc[12:]}'
    
    if New_fileName in sorted(os.listdir(output_folder), reverse=True):
        if 'sadeh' in pd.read_csv(os.path.join(output_folder,New_fileName)).columns:
            print(f'{New_fileName} already exist')
            if len(pd.read_csv(os.path.join(output_folder,New_fileName)).ZCY) > 12*60:
                print(f'{New_fileName} a acceptable size file exist')
                continue
            else:
                os.remove(os.path.join(output_folder,New_fileName))
                
    if f'{subject_id[j]}' in path_acc and FileName.endswith('.csv'):
        print(f'processing zcy {file_date}')
        
        columns_needed = ['date', 'y']  # Add any other necessary columns here
        data = pd.read_csv(FileName, usecols=columns_needed)

        data['date'] = pd.to_datetime(data['date'], utc=True, errors='coerce')
        data = data.dropna(subset=['date'])

        target_timezone = pytz.timezone(tz_str) 

        data['date'] = data['date'].dt.tz_convert(target_timezone)
        
        data.set_index('date', inplace=True)
        
        # Slice data to begin with nearest hour
        startTime = data.index[0]

        if startTime.minute < 15:
            startTimeNew = startTime.replace(microsecond=0, second=0, minute=15)
        elif startTime.minute >= 15 and startTime.minute < 30:
            startTimeNew = startTime.replace(microsecond=0, second=0, minute=30)
        elif startTime.minute >= 30 and startTime.minute < 45:
            startTimeNew = startTime.replace(microsecond=0, second=0, minute=45)
        else:
            if startTime.hour == 23:
                startTimeNew = startTime.replace(microsecond=0, second=0, minute=0, hour=0) + dt.timedelta(days=1)
            else:
                startTimeNew = startTime.replace(microsecond=0, second=0, minute=0, hour=startTime.hour + 1)

        data2 = data.loc[data.index >= startTimeNew]

        hb = 3
        lb = 0.25
        n = 2
        sf = 64

        Wc = [lb / (sf / 2), hb / (sf / 2)]

        b, a = signal.butter(n, Wc, 'bandpass')
        
        data2['y'] -= data2['y'].mean()
        data2['y'] = signal.lfilter(b, a, data2['y'])
        print("completed filtering")
        
        timeVec = data2.resample('5S').mean()
        print("completed resampling")

        for i in timeVec.index:
            mask = (data2.index <= i) & (data2.index > i - dt.timedelta(seconds=5))
            d = data2.loc[mask]

            if d.shape[0] == 320:
                y = d['y'].values
                y[np.abs(y) < 0.01] = 0

                Vec = np.ones_like(y)
                Vec[y < 0] = -1

                tmp = abs(np.sign(Vec[1:]) - np.sign(Vec[:-1])) * 0.5
                tmp = np.append(tmp[0], tmp)
                cs = np.cumsum(np.append(0, tmp))
                slct = np.arange(0, len(cs), 5 * sf)
                x3 = np.diff(cs[np.round(slct).astype(int)])
                timeVec.loc[i, 'ZCY'] = x3[0]
            else:
                timeVec.loc[i, 'ZCY'] = np.nan

        timeVec = timeVec.resample('60S').sum()
              
        if not timeVec.empty:

            timeVec['sadeh'] = sadeh_algorithm(timeVec['ZCY'], window_size=11)

            timeVec.to_csv(f'{output_folder}/{New_fileName}')

            print(f'finished processing zcy {New_fileName}')
