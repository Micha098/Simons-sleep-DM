import pandas as pd
import numpy as np
import os
import re
import sys
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import datetime
from sklearn import preprocessing
import subprocess

# Replace 'script.py' with the path to the script you want to call
subprocess.run(['python', 'python_files/Empatica_rawdata_bvp.py.py'])

i = int(sys.argv[1])
user_id = sys.argv[2]


input_folder = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/raw_data/{user_id}/acc/' #output folder
output_folder =  f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/measure_data/{user_id}/zcy/' 

files = sorted(os.listdir(input_folder), reverse= True)
#folders.remove('.DS_Store')
#load accelerometery datas
FileName = f'{input_folder}{files[i]}'
New_fileName = f'zcy{files[i][12:]}'

#if (New_fileName not in sorted(os.listdir(output_folder),reverse=True)) & (FileName.endswith('.csv')):
if (user_id in files[i]) & (FileName.endswith('.csv')):

    data = pd.read_csv(FileName,index_col=0)
    #try:
    data['date'] = data['time'].apply(lambda x: datetime.datetime.utcfromtimestamp(x))
    #except KeyError:
    #    try:
    #        data['date'] = data['timestamp'].apply(lambda x: datetime.datetime.utcfromtimestamp(x))
    #    except KeyError:
    #        pass

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
    Wc

    b,a = signal.butter(n, Wc, 'bandpass')
    # Calibrate the data by subtracting the mean
    data2 -= data2.mean()
    data2['y'] = signal.lfilter(b, a, data2['y'])
    print("completed filtering")
    timeVec = data2.resample('5S', convention = 'start').mean().drop(['time','x','y','z'],axis=1)
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

    timeVec = timeVec.resample('30S', convention = 'start').sum()

    #timeVec = timeVec.resample('60S', convention = 'start').sum()
    timeVec.to_csv(f'{output_folder}/{New_fileName}')

print(New_fileName)