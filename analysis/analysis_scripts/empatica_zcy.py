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


j =  int(sys.argv[1]) 

#target_date = sys.argv[2]
execute_preprocessing = True

subject_id = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['id'].tolist()
tzs_str = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['tz_str'].tolist()


input_folder = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/{subject_id[j]}/acc/' 
#output folder
output_folder =  f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/{subject_id[j]}/zcy/' 

if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

files = sorted(os.listdir(input_folder), reverse= True)
date_list = [re.search(r'\d{4}-\d{2}-\d{2}', file).group() for file in os.listdir(input_folder) if file.endswith(".csv") and file.startswith("empatica_acc") ]

date_list = sorted(date_list, reverse= True)
#load accelerometery datas

for datei, filei in zip(date_list,files):
    
    file_date = dt.datetime.strptime(datei, '%Y-%m-%d').date()
    print(f'found file: subject_{j} date {file_date}')
    path_acc = filei
# path_acc = None  # Initialize path_edf outside the loop

# for pathi in files:
#     try:
#         file_date = re.search(r'\d{4}-\d{2}-\d{2}', pathi).group()
#     except:
#         continue
#     if file_date:        
#         if file_date == target_date:
#             path_acc = pathi
#             file_date = dt.datetime.strptime(file_date, '%Y-%m-%d').date()
#             break

# if path_acc:

    FileName = f'{input_folder}{path_acc}'
    New_fileName = f'empatica_zcy{path_acc[12:]}'
    
    if (New_fileName in sorted(os.listdir(output_folder),reverse=True)):
        print(f'{New_fileName} already exist')

        if (os.path.getsize(output_folder+New_fileName) > 50000):
            print(f'{New_fileName} a accaptable size file exist')
            sys.exit()

    else:

        if (f'{subject_id[j]}' in path_acc) & (FileName.endswith('.csv')):
            print(f'processing zcy {file_date}')
            
            # Read the CSV file
            columns_needed = ['date', 'y']  # Add any other necessary columns here
            data = pd.read_csv(FileName, usecols=columns_needed)

            # Convert 'date' column to datetime, specifying the format directly
            data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S.%f')
            data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S.%f')

            data = data.set_index('date', drop = True)
            
            # slice data to begin with nearest hour
            startTime = data.index[0]
            #startTimeNew = startTime.replace(microsecond=540000, second=0, minute=0, hour=startTime.hour+1)
            if startTime.minute< 15:
                startTimeNew = startTime.replace(microsecond=0, second=0, minute=15, hour=startTime.hour)
            elif (startTime.minute>= 15 & startTime.minute < 30):
                startTimeNew = startTime.replace(microsecond=0, second=0, minute=30, hour=startTime.hour)
            elif (startTime.minute>= 30 & startTime.minute < 45):
                startTimeNew = startTime.replace(microsecond=0, second=0, minute=45, hour=startTime.hour)
            else:
                startTimeNew = startTime.replace(microsecond=0, second=0, minute=45, hour=startTime.hour+1)

            data2 = data.loc[data.index>=startTimeNew,]

            # set parameters (like GGIR)
            hb = 3
            lb = 0.25
            n = 2
            sf = 64

            Wc = np.zeros(2)
            Wc[0] = lb/(sf/2) 
            Wc[1] = hb/(sf/2)

            b,a = signal.butter(n, Wc, 'bandpass')
            # Calibrate the data by subtracting the mean
            data2 -= data2.mean()
            data2['y'] = signal.lfilter(b, a, data2['y'])
            print("completed filtering")
            timeVec = data2.resample('5S', convention = 'start').mean().drop(['y'],axis=1)
            print("completed resamplig")
            
            for i in timeVec.index:
                mask = (data2.index<=i) & (data2.index > i-datetime.timedelta(seconds=5))
                d = data2.loc[mask]

                if d.shape[0] == 320:
                    y = d['y'].values
                    #y=signal.lfilter(b,a,d['y'])
                    Ndat = len(y)
                    #change the values of y < 0.01 to 0
                    y[np.abs(y)<0.01]=0


                    # Create the vector of 1 and -1
                    Vec = np.ones_like(y)
                    Vec[y<0]=-1

                    tmp = abs(np.sign(Vec[1:Ndat])-np.sign(Vec[0:Ndat-1]))*0.5
                    tmp = np.append(tmp[0],tmp)
                    cs = np.cumsum(np.append(0,tmp))
                    slct = np.arange(0, len(cs), 5*sf)
                    x3 = np.diff(cs[np.round(slct)])
                    timeVec.loc[i,'ZCY']=x3[0]
                else:
                    timeVec.loc[i,'ZCY']=np.nan

            #timeVec = timeVec.resample('30S', convention = 'start').sum()

            timeVec = timeVec.resample('60S', convention = 'start').sum()
            timeVec.to_csv(f'{output_folder}/{New_fileName}')

            print(f'finished processing zcy {New_fileName}')
