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
import subprocess
import pandas as pd
import datetime as dt
from datetime import datetime
from datetime import timedelta
from datetime import time
import shutil
import pytz
from dateutil import parser as dt_parser  # Aliased to avoid conflicts
from utilities import project_data_dir
from eeg_sleep import EEGSleep
import argparse
import mne
import time as timer
subject_ids = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['id'].drop_duplicates().tolist()
subject_tzs = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['tz'].tolist()
subject_tzs_str = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['tz_str'].tolist()
output_dir = "/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/"


sleep_measures = pd.DataFrame()

sleep_measures_all= pd.DataFrame()

for user,tz_str in zip(subject_ids,subject_tzs_str):
    try:
        # tz_str = subject_tzs_str[i]
        zcy_all = pd.DataFrame()
        sleep_measures= pd.DataFrame()

        input_dir = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/{user}/zcy/'
        agg_dir = f'/mnt/home/mhacohen/ceph/agg_data/agg_emp_all_{user}.csv'

        for filename in sorted(os.listdir(input_dir)):
            if filename.startswith(f'empatica_zcy_{user}') and 'sadeh' in pd.read_csv(f'{input_dir}/{filename}').columns:
                dfi = pd.read_csv(f'{input_dir}/{filename}')
                zcy_all = pd.concat([dfi,zcy_all])[['date','ZCY','sadeh']]

        zcy_all.sort_values('date', inplace = True)  
        zcy_all.reset_index(drop = True, inplace = True) 

        agg_data = pd.read_csv(f'{agg_dir}').rename(columns = {'timestamp_iso':'date'})

        zcy_all = zcy_all.merge(agg_data[['date','wearing_detection_percentage','sleep_detection_stage']], how='left', on = 'date') 

        zcy_all.sort_values('date', inplace = True)
        zcy_all.reset_index(drop = True, inplace = True)
        zcy_all.loc[(zcy_all['sadeh'].isna()), ['sadeh']] = 0

        zcy_all.loc[(zcy_all['wearing_detection_percentage'] <= 50) | (zcy_all['wearing_detection_percentage'].isna()), ['sadeh', 'ZCY']] = np.nan

        target_timezone = pytz.timezone(tz_str) 
        
        zcy_all_copy = zcy_all.copy() # save a copy of the df in case of an empty one
        

        zcy_all['date'] = pd.to_datetime(zcy_all['date'], errors='coerce', utc=True)
        zcy_all['date'] = zcy_all['date'].dt.tz_convert(target_timezone)
        zcy_all = zcy_all.dropna(subset=['date'])
        
        zcy_all.set_index('date', inplace=True)

        # Resample the data to ensure every minute is included

        zcy_all.sort_values('date', inplace = True)  
        zcy_all = zcy_all[~zcy_all.index.duplicated(keep='first')]
       
        zcy_all.resample('T').asfreq()

        zcy_all.reset_index(inplace = True)

        if zcy_all.ZCY.dropna().empty: # different processing for users fo which resampling result in empyt df
            
            print(f'empty drop {user}')
            
            # give up on resampling process
            
            zcy_all = zcy_all_copy
            zcy_all = zcy_all.drop_duplicates('date')
            zcy_all.sort_values('date', inplace = True)  
            zcy_all.reset_index(drop = True, inplace = True)

        zcy_all.sort_values('date', inplace = True)  

        zcy_all.reset_index(inplace = True)
        zcy_all.drop_duplicates(inplace=True) 

        df = zcy_all.copy()

        # Intialize lists
        TST = []
        WASO = [] 
        SO = []
        FA = []
        results = []

        # Initialize the first search index
        start_index = 0

        while start_index < len(df) - 10:
            so_index = None
            fa_index = None

            # Search for sleep onset (so)
            for i in range(start_index, len(df)-10):
                if all(df.iloc[i:i+10]['sadeh'] == 1):
                    so_index = i
                    break

            if so_index is not None:
                # Look for FA starting from the end of the detected SO.
                # Find 2 hours of consecutive wakefullness
                for i in range(so_index + 10, so_index+(60*14)):
                    if all(df.iloc[i:i+120]['sadeh'].isin([0,np.nan])):
                        print(f'{i}-fa')
                        # Iterate backwards to find the last 10 consecutive sleep epochs
                        for j in range(i - 1, so_index + 9, -1):
                            if all(df.iloc[j-9:j+1]['sadeh'] == 1):
                                fa_index = j
                                break
                        if fa_index is None:
                            fa_index = i
                        break


                if fa_index is not None:
                    tst = (df.loc[so_index:fa_index, 'sadeh'] == 1).sum()
                    waso = df.loc[so_index:fa_index, 'sadeh'].isin([0,np.nan]).sum()
                    if (fa_index - so_index - waso) > 180: # check for short or fregmented sleep periods
                        SO.append(df['date'][so_index])
                        FA.append(df['date'][fa_index])
                        print(so_index)

                        TST.append(tst)
                        WASO.append(waso)
                        results.append({'TST': tst, 'WASO': waso, 'SO': SO[-1], 'FA': FA[-1]})
                        print(f'TST: {tst}, WASO: {waso} SO: {SO[-1]}, FA: {FA[-1]}')
                    # Update start_index to the index right after the current FA for the next cycle
                    start_index = fa_index + 1
                else:
                    # If no FA found, continue searching from 12h later
                    start_index = start_index + 60*3
                    print('artificaly added fa')

            else:
                # If no SO found, break out of the loop
                break

        sleep_measures_per_night = pd.DataFrame(results)

        if 'FA' in sleep_measures_per_night.columns:

            sleep_measures_per_night['FA'] = pd.to_datetime(sleep_measures_per_night['FA'])

            # Create a new column 'date' that extracts just the date part of the 'FA' datetime
            sleep_measures_per_night['date'] = sleep_measures_per_night['FA'].dt.date

            sleep_measures_per_night = sleep_measures_per_night[['date','TST','WASO','FA','SO']]#dfd.columns]

            os.makedirs(input_dir, exist_ok=True)

            sleep_measures_per_night['subject'] = user
        
            sleep_measures = pd.concat([sleep_measures,sleep_measures_per_night])
        
        sleep_measures.to_csv(f'{input_dir}/sadeh_nights_summary_{user}.csv')
        print(f'completed user {user}')  
    except Exception as e:
        print(f'{user}:{e}')
        
