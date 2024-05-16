from avro.datafile import DataFileReader
from avro.io import DatumReader
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
import json
import os
import glob 
import pytz
import numpy as np
import pandas as pd
import re
import scipy
import seaborn as sns
from scipy import signal
import ast
from scipy.stats import kendalltau, pearsonr, spearmanr,linregress
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.dates as mdates
from scipy.stats import ttest_ind

import datetime as dt
from datetime import timedelta
from datetime import time
import pytz 
from dateutil import parser
import subprocess
import warnings
warnings.filterwarnings("ignore")
import ast
import shutil
import zipfile

# Get dates per subject    
data_check = pd.read_csv('/mnt/home/mhacohen/Participants and devices - withings data check.csv')[['User ID','Starting date','End Date']]

df_id = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv').drop_duplicates()

df_id.drop_duplicates().sort_values('id', inplace = True)


data_check = data_check.dropna(subset=['Starting date', 'End Date'], how='all')
data_check['User ID'] = data_check['User ID'].str.replace('u', 'U')

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

## Process data for data share rep

def process_withings_shared_data(subject_id,tz,data_path,output_path, dates):
    try:
        input_folder_wit = glob.glob(f'{data_path}/**/*{subject_id}*/', recursive=True)[0]
    except Exception as e:
        print(f"Error processing data for subject_id {subject_id} {e}")
        return None
    data_folders = sorted([folder for folder in os.listdir(input_folder_wit) if folder.isdigit()], key=int, reverse=True)


    input_zip = None

    # Iterate over the folders from newest to oldest
    for data_folder in data_folders:
        input_folder_wit_current = os.path.join(input_folder_wit, data_folder)
        potential_zip_files = glob.glob(f'{input_folder_wit_current}/*data*.zip', recursive=True)

        for zip_path in potential_zip_files:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                if any("raw_bed_Pressure.csv" in file for file in zip_ref.namelist()):
                    input_zip = zip_path  # Take the first .zip file found

        break


    if input_zip:
        # Open the zip file
        with zipfile.ZipFile(input_zip, 'r') as zip_ref:
            # Extract all the contents into the directory
            zip_ref.extractall(input_folder_wit)

            for file in os.listdir(input_folder_wit):
                if file in ['raw_bed_Pressure.csv','raw_bed_HR RMS SD.csv',
                            'raw_bed_HR SD NN.csv','raw_bed_hr.csv','sleep.csv']:

                    if file.endswith('.csv'):
                        dfi = pd.read_csv(os.path.join(input_folder_wit,file))
                        if dfi.empty:
                            continue
                        if 'start' in dfi.columns:
                            date_columns = 'start'

                        elif 'date' in dfi.columns:
                            date_columns = 'date'

                        elif 'Date' in dfi.columns:
                            date_columns = 'Date'

                        elif 'from' in dfi.columns:
                            date_columns = ['from','to']

                            dfi['from'] = pd.to_datetime(dfi['from'],utc=True)
                            dfi['to'] = pd.to_datetime(dfi['to'],utc=True)

                            target_timezone = pytz.timezone(tz_str) 

                            dfi['from'] = dfi['from'].dt.tz_convert(target_timezone)
                            dfi['to'] = dfi['to'].dt.tz_convert(target_timezone)

                            dfi.to_csv(os.path.join(output_path,file), index = False)

                            continue


                        else:
                            dfi.to_csv(os.path.join(output_path,file), index = False)

                            continue

                    dfi.sort_values(date_columns, inplace = True)

                    dfi[date_columns] = pd.to_datetime(dfi[date_columns],utc=True)

                    dfi.drop_duplicates(inplace=True)

                    target_timezone = pytz.timezone(tz_str) 

                    dfi[date_columns] = dfi[date_columns].dt.tz_convert(target_timezone)

                    for col in [date_columns]: 
                            dfi['date_corrected'] = dfi[col].apply(lambda x: x if (pd.notna(x) and x.strftime('%Y-%m-%d') in dates) or
                                                                   ((x + timedelta(days=1)).strftime('%Y-%m-%d') in dates) else pd.NaT)
                    
                    dfi.dropna().to_csv(os.path.join(output_path,file), index = False)
                        
                        

data_path= f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/withings/'
subject_ids = pd.read_csv(f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv').drop('Unnamed: 0', axis= 1)['id']
tzs_str= pd.read_csv(f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv').drop('Unnamed: 0', axis= 1)['tz_str']


for user,tz_str, dates in zip(data_check['User ID'],data_check['tz_str'],data_check.dates):
    
    output_share_dir = f"/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/data_share/{user}/withings/"
    if not os.path.isdir(output_share_dir):
        os.makedirs(output_share_dir, exist_ok=True)

    process_withings_shared_data(user,tz_str,data_path,output_share_dir,dates)
    
    for file in os.listdir(output_share_dir):
        if file.startswith('withings'):
            os.remove(os.path.join(output_share_dir,file))
    print(user)

# Process data for study rep
    
def process_withings_data(subject_id,tz,data_path,dates):
        
        try:
            input_folder_wit = glob.glob(f'{data_path}/**/*{subject_id}*/', recursive=True)[0]
        except Exception as e:
            print(f"Error processing data for subject_id {subject_id} {e}")
            return None
        data_folders = sorted([folder for folder in os.listdir(input_folder_wit) if folder.isdigit()], key=int, reverse=True)


        input_zip = None

        # Iterate over the folders from newest to oldest
        for data_folder in data_folders:
            input_folder_wit_current = os.path.join(input_folder_wit, data_folder)
            potential_zip_files = glob.glob(f'{input_folder_wit_current}/*data*.zip', recursive=True)

            for zip_path in potential_zip_files:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    if any("raw_bed_Pressure.csv" in file for file in zip_ref.namelist()):
                        input_zip = zip_path  # Take the first .zip file found
                        print(zip_path)

            break
                        
        
        if input_zip:
            # Open the zip file
            with zipfile.ZipFile(input_zip, 'r') as zip_ref:
                # Extract all the contents into the directory
                zip_ref.extractall(input_folder_wit)
        
            # delete all other folders except the one that contains the extracted data
            # for item in os.listdir(input_folder_wit):
            #     item_path = os.path.join(input_folder_wit, item)
            #     # Check if it is a folder and not the data folder
            #     if os.path.isdir(item_path):
            #         shutil.rmtree(item_path)  # Deletes the folder and all its contents

            dfwith = pd.read_csv(f'{input_folder_wit}raw_bed_Pressure.csv')
            s_state = pd.read_csv(f'{input_folder_wit}raw_bed_sleep-state.csv')
            
            dfwith.start = pd.to_datetime(dfwith.start)
            
            dfwith.set_index((dfwith.start), drop=True, inplace=True)
            dfwith.drop(columns={'start'}, inplace=True)

            dfwith['duration'] = dfwith['duration'].apply(lambda x: ast.literal_eval(x))
            dfwith['value'] = dfwith['value'].apply(lambda x: ast.literal_eval(x))
            dfwith.sort_index(inplace=True)


            for i in range(len(dfwith)):
                start_times = [dfwith.index[i] + dt.timedelta(seconds=j * 60) for j in range(len(dfwith['duration'].iloc[i]))]
                values = dfwith['value'].iloc[i]

                if len(start_times) == len(values):
                    locals()[f'df_press_wit_{i}'] = pd.DataFrame({
                        'time': start_times,
                        'press_with': values
                    })
                else:
                    print('Error: start_times and values arrays have different lengths')

                if i > 0:
                    locals()[f'df_press_wit_{i}'] = pd.concat([locals()[f'df_press_wit_{i-1}'], locals()[f'df_press_wit_{i}']])

                if i == len(dfwith) - 1:
                    df_pressw = locals()[f'df_press_wit_{i}'].reset_index(drop=True)
                    df_pressw['date'] = df_pressw.time 
                    df_pressw.sort_index(inplace=True)

            #df_pressw.press_with = (df_pressw.press_with - np.mean(df_pressw.press_with)) / np.std(df_pressw.press_with)
            df_pressw['diff_press'] = abs(df_pressw['press_with'].diff())
            df_pressw['subject'] = subject_id

            # Process sleep state data
            s_state = pd.read_csv(f'{input_folder_wit}raw_bed_sleep-state.csv')
            s_state['start'] = pd.to_datetime(s_state['start'])
            
            s_state.set_index('start', inplace=True)
            s_state.sort_index(inplace=True)

            # Convert string representations of lists into actual lists
            s_state['duration'] = s_state['duration'].apply(ast.literal_eval)
            s_state['value'] = s_state['value'].apply(ast.literal_eval)

            df_s_state = pd.DataFrame()

            # Loop to expand each row into multiple rows based on duration and value
            for i, row in s_state.iterrows():
                start_times = [i + timedelta(seconds=j * 60) for j in range(len(row['duration']))]
                values = row['value']

                if len(start_times) == len(values):
                    df_twith = pd.DataFrame({
                        'time': start_times,
                        'sleep_state': values
                    })
                    df_s_state = pd.concat([df_s_state, df_twith])

            df_s_state.reset_index(drop=True, inplace=True)

            dfs = df_pressw.merge(df_s_state, on = 'time', how = 'left')
            
            # filter dates from data_check tables
            
            dfs['time'] = dfs['time'].apply(lambda x: x if (pd.notna(x) and x.strftime('%Y-%m-%d') in dates) or
                                           ((x + timedelta(days=1)).strftime('%Y-%m-%d') in dates) else pd.NaT)
        return dfs.dropna()  # Move this line outside the except block

def z_score_outlier_filter(data, threshold):
    mean_val = np.mean(data)
    std_val = np.std(data)
    filtered_data = np.where(np.abs(data - mean_val) <= threshold * std_val, data, np.nan)
    return np.nan_to_num(filtered_data)

