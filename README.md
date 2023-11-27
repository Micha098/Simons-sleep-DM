# Simons-sleep-DM
Data management scripts for Simons Sleep Project 
# Data Management for Sleep Study

## Overview
This repository contains scripts and files for managing data related to a sleep study. The data processing involves Slurm scripts for iteration over dates and subjects, Python scripts for data processing, and a Jupyter notebook for generating visualizations.

# 1. Slurm Scripts

## 1.1 Iteration Over Dates (i.e. `slurm_files/slurm_zcy.sh`)

#!/bin/bash
#SBATCH --time 1:00:00
#SBATCH --job-name=Sleep_study
#SBATCH --output=ceph/emp_zcy_%a.out
#SBATCH --mem 20GB

### Loop over specified dates
for target_date in {2023-11-17,2023-11-18,2023-11-19,2023-11-20,2023-11-21,2023-11-22,2023-11-23,2023-11-24,2023-11-25,2023-11-26,2023-11-27,2023-11-28}; do
    # Call your Python script and pass the subject ID and date as arguments
    sbatch slurm_files/init_conda.sh
    sbatch --export=TARGET_DATE=$target_date slurm_files/slurm_zcy_job.sh
done

## 1.2 Iteration Over Subjects (i.e. slurm_files/slurm_zcy_job.sh)
#!/bin/bash
#SBATCH --time 1:00:00
#SBATCH --job-name=Sleep_study
#SBATCH --output=ceph/dreem_job%a.out
#SBATCH --array=1-10   # Number of tasks/subjects
#SBATCH --mem 20GB

### Load the necessary modules or activate the virtual environment if required
source slurm_files/init_conda.sh

### Change to the directory containing your Python script
cd /mnt/home/mhacohen/python_files

### Call your Python script and pass the subject ID as an argument
python empatica_zcy.py $SLURM_ARRAY_TASK_ID $TARGET_DATE

# 2. Python Files

## i.e. empatica_zcy.py
This Python script processes data for the Dreem sleep study. It reads Empatica accelerometer data, performs preprocessing, and generates activity counts.

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



i = int(sys.argv[1]) -1 
date = sys.argv[2]
execute_preprocessing = True

subject_id = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['id'].tolist()

#define path, folders, user 
participant_data_path = '/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/aws_data/1/1/participant_data/' # path to the folder that contains folder for each date
output_folder = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/raw_data/{subject_id[i]}/acc/' #output folder


# accelerometer data 
dfs = pd.DataFrame()
dates = sorted(os.listdir(participant_data_path), reverse=True) #all date-folders available 
#dates.remove('.DS_Store')
filename = f'empatica_acc_{subject_id[i]}_'+date+'.csv'

if (filename in sorted(os.listdir(output_folder),reverse=True)):
    
    if (os.path.getsize(output_folder+filename) > 250000000):
        print(f'{filename} already exist')
        execute_preprocessing = False

elif execute_preprocessing:
    print(date)
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

                acc = data["rawData"]["accelerometer"] #access specific metric 
                startSeconds = acc["timestampStart"] / 1000000 # convert timestamp to seconds
                timeSeconds = list(range(0,len(acc['x'])))
                if acc["samplingFrequency"] == 0:
                    acc["samplingFrequency"] = 64;
                timeUNIX = [t/acc["samplingFrequency"]+startSeconds for t in timeSeconds]
                delta_physical = acc["imuParams"]["physicalMax"] - acc["imuParams"]["physicalMin"]
                delta_digital = acc["imuParams"]["digitalMax"] - acc["imuParams"]["digitalMin"]
                acc['x'] = [val*delta_physical/delta_digital for val in acc["x"]]
                acc['y'] = [val*delta_physical/delta_digital for val in acc["y"]]
                acc['z'] = [val*delta_physical/delta_digital for val in acc["z"]]

                df_acTot = pd.concat([pd.DataFrame(timeUNIX), pd.DataFrame(acc['x']),pd.DataFrame(acc['y']),pd.DataFrame(acc['z'])],axis = 1)

                if not df_acTot.empty:
                    df_acTot.columns = ['time','x','y','z']
                    dfs = pd.concat([dfs,df_acTot])
            dfs=dfs.reset_index(drop = True)
            dfs.to_csv(output_folder+f'empatica_acc_{subject_id[i]}_'+date+'.csv')
            dfs = pd.DataFrame()
            print('finished preprocessing '+date)
    else:
        print(f'subfolfer {date} empty')


input_folder = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/raw_data/{subject_id[i]}/acc/' #output folder
output_folder =  f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/measure_data/{subject_id[i]}/zcy/' 


files = sorted(os.listdir(input_folder), reverse= True)
#folders.remove('.DS_Store')
#load accelerometery datas

target_date = sys.argv[2]

path_acc = None  # Initialize path_edf outside the loop

for pathi in files:
    try:
        file_date = re.search(r'\d{4}-\d{2}-\d{2}', pathi).group()
        print(file_date)
    except:
        continue
    if file_date:        
        if file_date == target_date:
            path_acc = pathi
            file_date = dt.datetime.strptime(file_date, '%Y-%m-%d').date()
            break

if path_acc:

    FileName = f'{input_folder}{path_acc}'
    New_fileName = f'zcy{path_acc[12:]}'
    
    if (New_fileName in sorted(os.listdir(output_folder),reverse=True)):
        print(f'{New_fileName} already exist')

        if (os.path.getsize(output_folder+New_fileName) > 50000):
            print(f'{New_fileName} a accaptable size file exist')
            sys.exit()

    else:

        if (f'{subject_id[i]}' in path_acc) & (FileName.endswith('.csv')):

            data = pd.read_csv(FileName,index_col=0)
            #try:
            data['date'] = data['time'].apply(lambda x: dt.datetime.utcfromtimestamp(x))
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

            timeVec = data2.resample('5S', convention = 'start').mean().drop(['time','x','y','z'],axis=1)
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

            print(f'finished processing {New_fileName}')
# 3. Jupyter Notebook

## 3.1 Daily_test_script.ipynb
This Jupyter notebook generates visualizations from the output of the Python scripts.

# Usage

Run the Slurm scripts to iterate over dates and subjects.
The Python scripts process the data and generate output files.
Use the Jupyter notebook to visualize the results.

