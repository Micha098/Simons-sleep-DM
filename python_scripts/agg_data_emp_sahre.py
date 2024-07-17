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
import sys
import datetime as dt
from datetime import timedelta
from datetime import time
import subprocess
import warnings
warnings.filterwarnings("ignore")
import ast
import shutil


dates = []
i = int(sys.argv[1])

agg_emp_all = pd.DataFrame()
df_temp = pd.DataFrame()

aggregated_data_path = None
    
users = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['id'].tolist()
tzs = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['tz'].tolist()
tzs_str = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['tz_str'].tolist()

measure = ['pulse-rate', 'prv','activity-counts', 'sleep','step','respiratory','wearing-detection']
participant_data_path = '/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/aws_data/1/1/participant_data/'



data_check = pd.read_csv('/mnt/home/mhacohen/Participants and devices - embrace data check.csv')[['User ID','Starting date','End Date']]

data_check.sort_values('Starting date', inplace = True)

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
# data_check.dropna(subset=['dates'], inplace=True)

dates = data_check['dates'].tolist()

user = users[i]
tz_str = tzs_str[i]
dates = dates[i]

print(dates)
    
for d in dates:
    print(d)
    for foldername in os.listdir(os.path.join(participant_data_path,d)):

        if f'{user}-' in foldername:
            try:
                aggregated_data_path= glob.glob(f'{participant_data_path}/{d}/*U{user}*/digital_biomarkers/aggregated_per_minute/')[0]
            except:
                continue
    
        if user in [104,150,151]: # correct mistake in empatica usernames lacking U

            for foldername in os.listdir(os.path.join(participant_data_path,d)):
                if f'{user}-' in foldername:
                    try:
                        aggregated_data_path= glob.glob(f'{participant_data_path}/{d}/*{user}-3Y*/digital_biomarkers/aggregated_per_minute/')[0]
                        print(foldername)
                    except:
                        continue

        if user == 387 and d in ['2024-07-04', '2024-07-05', '2024-07-06','2024-07-07', '2024-07-08']: # correct mistake in user 187/387            
            for foldername in os.listdir(os.path.join(participant_data_path,d)):
                if '187-' in foldername:
                    try:
                        aggregated_data_path= glob.glob(f'{participant_data_path}/{d}/*U187*/digital_biomarkers/aggregated_per_minute/')[0]
                    except:
                        continue

                        
        if aggregated_data_path and len(os.listdir(aggregated_data_path)) > 1:
            
            wear_path = glob.glob(os.path.join(aggregated_data_path, f'*{measure[6]}*'))[0]
            agg_wear = pd.read_csv(wear_path).iloc[:,3]
            agg_wear['subject'] = user

            try:
                hrv_path = glob.glob(os.path.join(aggregated_data_path, f'*{measure[1]}*'))[0]
                agg_hrv = pd.read_csv(hrv_path).iloc[:,3]
                agg_hrv['subject'] = user
                rr_path = glob.glob(os.path.join(aggregated_data_path, f'*{measure[5]}*'))[0]
                agg_rr = pd.read_csv(rr_path).iloc[:,3]
                agg_rr['subject'] = user

            except Exception as e:
                agg_hrv = pd.DataFrame()
                agg_rr = pd.DataFrame()

            try:
                sleep_path = glob.glob(os.path.join(aggregated_data_path, f'*{measure[3]}*'))[0]
                agg_sleep = pd.read_csv(sleep_path).iloc[:,3]
                agg_sleep['subject'] = user

            except Exception as e:
                agg_sleep = pd.DataFrame()

            hr_path = glob.glob(os.path.join(aggregated_data_path, f'*{measure[0]}*'))[0]
            ac_path = glob.glob(os.path.join(aggregated_data_path, f'*{measure[2]}*'))[0]
            step_path = glob.glob(os.path.join(aggregated_data_path, f'*{measure[4]}*'))[0]


            agg_hr = pd.read_csv(hr_path)
            agg_ac = pd.read_csv(ac_path).iloc[:,3]
            agg_step = pd.read_csv(step_path).iloc[:,3]

            # Add a new column 'subject' with the subject ID
            agg_hr['subject'] = user
            agg_ac['subject'] = user
            agg_step['subject'] = user
            
            df_temp = pd.concat([agg_hr,agg_hrv, agg_ac,agg_sleep,agg_step,agg_rr,agg_wear], axis = 1)
            df_temp.drop('subject',axis = 0, inplace = True)

            agg_emp_all = pd.concat([agg_emp_all,df_temp]).reset_index(drop=True)



if d == sorted(dates)[-1] and not agg_emp_all.empty:


    agg_emp_all.reset_index(inplace = True, drop = True)
    agg_emp_all.sort_index(inplace=True)
    agg_emp_all.drop_duplicates('timestamp_unix', keep = 'first')
    agg_emp_all['timestamp_iso'] = pd.to_datetime(agg_emp_all['timestamp_iso'])
    target_timezone = pytz.timezone(tz_str) 
    
    print(f'{user} {target_timezone}')
    agg_emp_all['timestamp_iso'] = agg_emp_all['timestamp_iso'].dt.tz_convert(target_timezone)

    target_path = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/data_share/{user}/empatica/summarized_data'

    
    if os.path.isdir(target_path):
        shutil.rmtree(target_path)
   
    if not os.path.isdir(target_path) or not os.path.isdir(target_path_share):
        os.makedirs(target_path,exist_ok=True)
        os.makedirs(target_path_share,exist_ok=True)


    shutil.rmtree(target_path_share)


    if not os.path.isdir(target_path) or not os.path.isdir(target_path_share):
        os.makedirs(target_path,exist_ok=True)
        os.makedirs(target_path_share,exist_ok=True)


    if os.path.isdir(target_path):
        # Iterate over all files and directories within 'target_path'
        for item in os.listdir(target_path):
            # Construct full path to item
            item_path = os.path.join(target_path, item)
            # Check if the item is a file and starts with the specified prefix
            if os.path.isfile(item_path) and item.startswith('empatica_measures'):
                # Delete the file
                os.remove(item_path)


if not agg_emp_all.empty and 'timestamp_iso' in agg_emp_all.columns:

    for datei in agg_emp_all['timestamp_iso'].dt.date.unique():

        new_filename = f'empatica_measures_{user}_{datei}.csv'
        agg_daily = agg_emp_all[agg_emp_all.timestamp_iso.dt.date == datei]

        agg_daily.to_csv(os.path.join(target_path,new_filename), index =False)
        
else:
    print(f'error {agg_emp_all.columns}')
