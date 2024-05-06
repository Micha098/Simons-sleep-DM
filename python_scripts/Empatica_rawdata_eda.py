from avro.datafile import DataFileReader
from avro.io import DatumReader
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime, timedelta
import json
import os
import shutil
import sys
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


i = int(sys.argv[1])
    
data_check = pd.read_csv('/mnt/home/mhacohen/Participants and devices - embrace data check.csv')[['User ID','Starting date','End Date']]

df_id = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv').drop_duplicates()

df_id.drop_duplicates().sort_values('id', inplace = True)


data_check = data_check.dropna(subset=['Starting date', 'End Date'], how='all')


data_check.loc[data_check['End Date'].isna(), 'End Date'] = dt.date.today()
data_check['Starting date'] = pd.to_datetime(data_check['Starting date'])
data_check['End Date'] = pd.to_datetime(data_check['End Date'], errors='coerce')

data_check.dropna(subset=['Starting date', 'End Date'], inplace=True)

def get_date_range(row):
    return pd.date_range(start=row['Starting date'], end=row['End Date']).date.tolist()

# Apply the function across the DataFrame
data_check['dates'] = data_check.apply(get_date_range, axis=1)
data_check['dates'] = data_check['dates'].apply(
    lambda x: [date.strftime('%Y-%m-%d') for date in x])

data_check.sort_values('User ID', inplace=True)
data_check.reset_index(inplace = True, drop = True)
data_check['User ID'] = pd.to_numeric(data_check['User ID'].str.replace('U', '', regex=True))
data_check = data_check.merge(df_id.rename(columns = {'id':'User ID'}), on = 'User ID', how = 'right')

subject_id = data_check['User ID'].tolist()
tzs_str = data_check['tz_str'].tolist()
data_check.dropna(subset=['dates'], inplace=True)

# Function to process and save data for adjusted tz
def process_and_save_adjusted_days(subject_id,tzs_str,output_folder,shared_data_folder,file1, file2=None):
    df1 = pd.read_csv(file1).rename(columns = {'timestamp':'time'})
    
    if file2:
        
        df2 = pd.read_csv(file2).rename(columns = {'timestamp':'time'})
        df_combined = pd.concat([df1, df2], ignore_index=True)
    else:
        df_combined = df1

    df_combined.drop_duplicates(subset='time', keep='first', inplace=True)

    #df_combined['date'] = pd.to_datetime(df_combined['time'], unit='s').dt.tz_localize(None)
    df_combined['date'] = pd.to_datetime(df_combined['time'], unit='s').dt.tz_localize('UTC')

    target_timezone = pytz.timezone(tzs_str) 

    print(f'{str(subject_id)} {target_timezone}')
    
    df_combined['date'] = df_combined['date'].dt.tz_convert(target_timezone)
    
    df_combined['day'] = df_combined['date'].dt.date

    for datei in df_combined['day'].unique():
        new_filename = f'empatica_eda_{subject_id}_{datei}.csv'
        df_combined[df_combined['day'] == datei].to_csv(os.path.join(output_folder, new_filename), index=False)
        df_combined[df_combined.day == datei].to_csv(os.path.join(shared_data_folder, new_filename), index=False)

        print(f'Adjusted data for {datei} saved.')

# def eda_raw_data(i):
    
#define path, folders, users 
participant_data_path = '/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/aws_data/1/1/participant_data/' # path to the folder that contains folder for each date
tz_temp = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/{subject_id[i]}/eda/tz_temp' #output folder

output_folder = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/{subject_id[i]}/eda/' #output folder

shared_data_folder = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/data_share/{subject_id[i]}/empatica/raw_data/eda/' #output folder

if os.path.isdir(tz_temp):
    shutil.rmtree(tz_temp)
    
shutil.rmtree(output_folder)
shutil.rmtree(shared_data_folder)

if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

if not os.path.isdir(shared_data_folder):
    os.makedirs(shared_data_folder)

if not os.path.isdir(tz_temp):
        os.makedirs(tz_temp)

        
# eda data 
def eda_raw_data(i):

    dfs = []

    dates = data_check.dates[i]
    print(dates)
    for date in dates: 

        try:

            if (f'empatica_eda_{subject_id[i]}_{date}.csv') in sorted(os.listdir(tz_temp),reverse=True):         
                print('file already processed')
                continue

    #             # elif (f'empatica_eda_{subject_id[i]}_{date}.csv') not in sorted(os.listdir(tz_temp),reverse=True) and ((f'empatica_eda_{subject_id[i]}_{date}.csv') not in sorted(os.listdir(output_folder),reverse=True)):   
    #             else:

            date_folder = os.path.join(participant_data_path+date) # list folders (for each user) within the date-folde
            for name_folder in os.listdir(date_folder):
                if (f'{subject_id[i]}-') in name_folder:
                    subfolder = os.path.join(date_folder, name_folder, 'raw_data', 'v6')
                    if  subfolder != []:
                        print(f'folder  {date}')
                        if os.path.isdir(subfolder):
                            files = os.listdir(subfolder) #list of avro files
                            files = np.sort(files).tolist() # rearrange files in a chronological manner
                            for ff in files: #loop through files to read and store data
                                avro_file = os.path.join(subfolder,ff)
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
                                    eda["samplingFrequency"] = 4;
                                timeUNIX = [t/eda["samplingFrequency"]+startSeconds for t in timeSeconds]
                                df_eda = pd.DataFrame({'time': timeUNIX, 'eda': eda['values']})

                                if not df_eda.empty:
                                    dfs.append(df_eda)

                            if dfs:

                                daily_df = pd.concat(dfs, ignore_index=True)
                                daily_filename = f'{tz_temp}/empatica_eda_{subject_id[i]}_{date}.csv'
                                daily_df.to_csv(daily_filename)
                                print(f'Data for {date} saved.')
                                dfs =[]
                                print(f'{date}')

        except Exception as e:
            print(e)
            continue

    day_files = sorted([f for f in os.listdir(tz_temp) if f.startswith('empatica_acc_') and f.endswith('.csv') and f not in os.listdir(output_folder)])
    print(day_files)
    if len(day_files) > 0:

        # Process each file, considering it and the next one for overlapping data
        for j, file in enumerate(day_files[:-1]):  # Exclude the last file for this loop
            process_and_save_adjusted_days(subject_id[i],tzs_str[i],output_folder,shared_data_folder,os.path.join(tz_temp, file), os.path.join(tz_temp, day_files[j+1]))

    if len(day_files) > 0:

        # Process the last file separately since it has no next file to combine with
        process_and_save_adjusted_days(subject_id[i],tzs_str[i],output_folder,shared_data_folder,os.path.join(tz_temp, day_files[-1]),None)
    # shutil.rmtree(tz_temp)
    print('Cleared tz_temp folder')

eda_raw_data(i)