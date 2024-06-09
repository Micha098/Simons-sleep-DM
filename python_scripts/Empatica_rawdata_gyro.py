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

i= int(sys.argv[1])

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

    print(f'{subject_id} {target_timezone}')
    
    df_combined['date'] = df_combined['date'].dt.tz_convert(target_timezone)
    
    df_combined['day'] = df_combined['date'].dt.date

    min_date = df_combined['day'].min()  # Find the minimum date
       
    # Filter out the minimum day
    filtered_df = df_combined[df_combined['day'] != min_date]

    # Save each day's data other than the minimum date
    for datei in filtered_df['day'].unique():
        new_filename = f'empatica_gyro_{subject_id}_{datei}.csv'
        filtered_df[filtered_df['day'] == datei].to_csv(os.path.join(output_folder, new_filename), index=False)
        filtered_df[filtered_df['day'] == datei].to_csv(os.path.join(shared_data_folder, new_filename), index=False)
        print(f'Adjusted data for {datei} saved.')

def gyro_raw_data(i):
    subject_id = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['id'].tolist()
    tzs_str = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['tz_str'].tolist()

    #define path, folders, uzaser 
    participant_data_path = '/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/aws_data/1/1/participant_data/' # path to the folder that contains folder for each date
    tz_temp = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/{subject_id[i]}/gyro/tz_temp' #output folder

    output_folder = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/{subject_id[i]}/gyro/' #output folder
    shared_data_folder = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/data_share/{subject_id[i]}/empatica/raw_data/gyro/' #output folder


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
    # gyro data 

    dfs = []
    dates = data_check.dates[i]
    print(dates)
    
    for date in dates: 
        try:

            if (f'empatica_gyro_{subject_id[i]}_{date}.csv') in sorted(os.listdir(tz_temp),reverse=True):         
                print('file already processed')
                continue

            # elif (f'empatica_gyro_{subject_id[i]}_{date}.csv') not in sorted(os.listdir(tz_temp),reverse=True) and ((f'empatica_gyro_{subject_id[i]}_{date}.csv') not in sorted(os.listdir(output_folder),reverse=True)):   
            date_folder = os.path.join(participant_data_path+date) # list folders (for each user) within the date-folde
            for name_folder in os.listdir(date_folder):
                if (f'{subject_id[i]}-') in name_folder:
                    print(name_folder)
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

                                gyro = data['rawData']['gyroscope']
                                print(gyro)
                                startSeconds = gyro["timestampStart"] / 1000000
                                timeSeconds = list(range(0,len(gyro['x'])))
                                #datetime_timetemp = [datetime.utcfromtimestamp(x) for x in timeUNIXtemp]
                                if gyro["samplingFrequency"] == 0:
                                    gyro["samplingFrequency"] = 64;
                                timeUNIX = [t/gyro["samplingFrequency"]+startSeconds for t in timeSeconds]

                                delta_physical = gyro["imuParams"]["physicalMax"] - gyro["imuParams"]["physicalMin"]
                                delta_digital = gyro["imuParams"]["digitalMax"] - gyro["imuParams"]["digitalMin"]
                                x_dps = [val * delta_physical / delta_digital for val in gyro["x"]]
                                y_dps = [val * delta_physical / delta_digital for val in gyro["y"]]
                                z_dps = [val * delta_physical / delta_digital for val in gyro["z"]]

                                df_gyroTot = pd.DataFrame({'time': timeUNIX, 'x_dps': gyro['x'], 'y_dps': gyro['y'], 'z_dps': gyro['z']})

                                if not df_gyroTot.empty:
                                    dfs.append(df_gyroTot)

                            if dfs:

                                daily_df = pd.concat(dfs, ignore_index=True)
                                daily_filename = f'{tz_temp}/empatica_gyro_{subject_id[i]}_{date}.csv'
                                daily_df.to_csv(daily_filename)
                                print(f'Data for {date} saved.')
                                dfs =[]
                                print(f'{date}')

        except Exception as e:
            print(e)
            continue


    day_files = sorted([f for f in os.listdir(tz_temp) if f.startswith('empatica_gyro_') and f.endswith('.csv') and (f not in os.listdir(output_folder))])
    
    if len(day_files) > 0:

        # Process each file, considering it and the next one for overlapping data
        for j, file in enumerate(day_files[:-1]):  # Exclude the last file for this loop
            process_and_save_adjusted_days(subject_id[i],tzs_str[i],output_folder,shared_data_folder,os.path.join(tz_temp, file), os.path.join(tz_temp, day_files[j+1]))
            print(file)
   
    if len(day_files) > 0:

        # Process the last file separately since it has no next file to combine with
        process_and_save_adjusted_days(subject_id[i],tzs_str[i],output_folder,shared_data_folder,os.path.join(tz_temp, day_files[-1]),None)
    # shutil.rmtree(tz_temp)
    print('Cleared tz_temp folder')

gyro_raw_data(i)
