#from avro.datafile import DataFileReader
#from avro.io import DatumReader
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


dates = []
i = int(sys.argv[1])

agg_emp_all = pd.DataFrame()
df_temp = pd.DataFrame()

aggregated_data_path = None
    
users = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['id'].tolist()
tzs = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['tz'].tolist()
tzs_str = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['tz_str'].tolist()

user = users[i]
tz = tzs[i]
tz_str = tzs_str[i]


measure = ['pulse-rate', 'activity-counts', 'temperature', 'sleep','prv','eda']
participant_data_path = '/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/aws_data/1/1/participant_data/'

files = [f for f in os.listdir(participant_data_path) if (f.startswith(r'202'))]
for fb in files:

    date = re.search(r"\d{4}-\d{2}-\d{2}", fb).group()
    date = dt.datetime.strptime(date, '%Y-%m-%d')
    date = date.strftime("%Y-%m-%d")
    dates.append(date)

for d in sorted(dates):
    for foldername in os.listdir(os.path.join(participant_data_path,d)):
        if f'U{user}' in foldername:
                                     
            aggregated_data_path= glob.glob(f'{participant_data_path}/{d}/*U{user}*/digital_biomarkers/aggregated_per_minute/')[0]

        if user in [104,150,151]: # correct mistake in empatica usernames lacking U

            for foldername in os.listdir(os.path.join(participant_data_path,d)):
                if f'{user}-' in foldername:

                    aggregated_data_path= glob.glob(f'{participant_data_path}/{d}/*{user}-3Y*/digital_biomarkers/aggregated_per_minute/')[0]
                    print(foldername)

    
        try:
            hr_path = glob.glob(os.path.join(aggregated_data_path, f'*{measure[0]}*'))[0]
            ac_path = glob.glob(os.path.join(aggregated_data_path, f'*{measure[1]}*'))[0]
            temp_path = glob.glob(os.path.join(aggregated_data_path, f'*{measure[2]}*'))[0]
            sleep_path = glob.glob(os.path.join(aggregated_data_path, f'*{measure[3]}*'))[0]
            hrv_path = glob.glob(os.path.join(aggregated_data_path, f'*{measure[4]}*'))[0]
            eda_path = glob.glob(os.path.join(aggregated_data_path, f'*{measure[5]}*'))[0]

            agg_hr = pd.read_csv(hr_path)
            agg_ac = pd.read_csv(ac_path).iloc[:,3]
            agg_temp = pd.read_csv(temp_path).iloc[:,3]
            agg_sleep = pd.read_csv(sleep_path).iloc[:,3]
            agg_hrv = pd.read_csv(hrv_path).iloc[:,3]
            agg_eda = pd.read_csv(eda_path).iloc[:,3]

            # Add a new column 'subject' with the subject ID
            agg_hr['subject'] = user
            agg_ac['subject'] = user
            agg_temp['subject'] = user
            agg_sleep['subject'] = user
            agg_hrv['subject'] = user
            agg_eda['subject'] = user

            df_temp = pd.concat([agg_hr,agg_hrv, agg_ac, agg_temp,agg_sleep,agg_eda], axis = 1)
            df_temp.drop('subject',axis = 0, inplace = True)

        except Exception as e:
            continue 

    agg_emp_all = pd.concat([agg_emp_all,df_temp]).reset_index(drop=True)



if d == sorted(dates)[-1] and not agg_emp_all.empty:


    agg_emp_all.reset_index(inplace = True, drop = True)
    agg_emp_all.sort_index(inplace=True)
    agg_emp_all.drop_duplicates('timestamp_unix', keep = 'first')
    agg_emp_all['timestamp_iso'] = pd.to_datetime(agg_emp_all['timestamp_iso'])
    target_timezone = pytz.timezone(tz_str) 
    
    print(f'{user} {target_timezone}')
    agg_emp_all['timestamp_iso'] = agg_emp_all['timestamp_iso'].dt.tz_convert(target_timezone)

    target_path = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/{user}/empatica_measures'

    if not os.path.isdir(target_path):
        os.makedirs(target_path)


    if os.path.isdir(target_path):
        # Iterate over all files and directories within 'target_path'
        for item in os.listdir(target_path):
            # Construct full path to item
            item_path = os.path.join(target_path, item)
            # Check if the item is a file and starts with the specified prefix
            if os.path.isfile(item_path) and item.startswith('empatica_measures'):
                # Delete the file
                os.remove(item_path)





for datei in agg_emp_all.timestamp_iso.dt.date.unique():

    new_filename = f'empatica_measures_{user}_{datei}.csv'
    agg_daily = agg_emp_all[agg_emp_all.timestamp_iso.dt.date == datei]

    agg_daily.to_csv(os.path.join(target_path,new_filename), index =False)

#agg_emp_all = pd.concat([agg_emp_all,pd.read_csv('/mnt/home/mhacohen/ceph/agg_emp_all.csv').reset_index(drop=True)])
agg_emp_all.to_csv(f'/mnt/home/mhacohen/ceph/agg_data/agg_emp_all_{user}.csv', index =False)
