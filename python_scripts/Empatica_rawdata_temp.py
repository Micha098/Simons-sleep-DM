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

    print(f'{subject_id} {target_timezone}')
    
    df_combined['date'] = df_combined['date'].dt.tz_convert(target_timezone)
    df_combined['day'] = df_combined['date'].dt.date

    for datei in df_combined['day'].unique():
        new_filename = f'{output_folder}/empatica_temp_{subject_id}_{datei}.csv'
        df_combined[df_combined['day'] == datei].to_csv(os.path.join(output_folder, new_filename), index=False)
        df_combined[df_combined['day'] == datei].to_csv(os.path.join(shared_data_folder, new_filename), index=False)

        print(f'Adjusted data for {datei} saved.')

def temp_raw_data(i):
    subject_id = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['id'].tolist()
    tzs_str = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['tz_str'].tolist()

    #define path, folders, uzaser 
    participant_data_path = '/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/aws_data/1/1/participant_data/' # path to the folder that contains folder for each date
    tz_temp = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/{subject_id[i]}/temp/tz_temp' #output folder

    output_folder = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/{subject_id[i]}/temperature/' #output folder
    shared_data_folder = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/data_share/{subject_id[i]}/empatica/raw_data/temperature/' #output folder


    if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
    if not os.path.isdir(tz_temp):
            os.makedirs(tz_temp)
    # temprature data 

    dfs = []
    dates = sorted(os.listdir(participant_data_path), reverse=True) #all date-folders available 
    dates.remove('.DS_Store')
    for date in [d for d in sorted(dates) if pd.to_datetime(d).date() > pd.to_datetime('2023-10-30').date()]:# date of starting the sleep study pilot

        if True: #f'empatica_temp_{subject_id[i]}_{date}.csv' not in sorted(os.listdir(output_folder),reverse=True) or (subject_id[i] in [133,333,310,110,147,347,115,315,140,340,158,358,159,359,161,361]):
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

                                temp = data["rawData"]["temperature"] #access specific metric 
                                if len(data["rawData"]["temperature"]['values']) > 0:

                                    startSeconds = temp["timestampStart"] / 1000000 # convert timestamp to seconds
                                    timeSeconds = list(range(0,len(temp['values'])))
                                    timeUNIX = [t/temp["samplingFrequency"]+startSeconds for t in timeSeconds]
                                    datetime_time = [datetime.utcfromtimestamp(x) for x in timeUNIX]

                                    df_tempTot = pd.DataFrame({'time': timeUNIX, 'temp': temp['values']})

                                    if not df_tempTot.empty:
                                        dfs.append(df_tempTot)

                            if dfs:

                                daily_df = pd.concat(dfs, ignore_index=True)
                                daily_filename = f'{tz_temp}/empatica_temp_{subject_id[i]}_{date}.csv'
                                daily_df.to_csv(daily_filename)
                                print(f'Data for {date} saved.')
                                dfs =[]
                                print(f'{date}')


        else:
            print('file already exists')



    day_files = sorted([f for f in os.listdir(tz_temp) if f.startswith('empatica_temp_') and f.endswith('.csv')])
    if len(day_files) > 0:

        # Process each file, considering it and the next one for overlapping data
        for j, file in enumerate(day_files[:-1]):  # Exclude the last file for this loop
            process_and_save_adjusted_days(subject_id[i],tzs_str[i],output_folder,shared_data_folder,os.path.join(tz_temp, file), os.path.join(tz_temp, day_files[j+1]))
    if len(day_files) > 0:

        # Process the last file separately since it has no next file to combine with
        process_and_save_adjusted_days(subject_id[i],tzs_str[i],output_folder,shared_data_folder,os.path.join(tz_temp, day_files[-1]),None)
    shutil.rmtree(tz_temp)
    print('Cleared tz_temp folder')

