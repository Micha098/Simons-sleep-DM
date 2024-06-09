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
os.chdir('/mnt/home/mhacohen/python_files/')
# from dreem_summ_data import summary_data_dreem

subject_ids = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['id'].tolist()
subject_tzs = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['tz'].tolist()
subject_tzs_str = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['tz_str'].tolist()

output_dir = "/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/"
summary_dir = "dreem_reports/"
def calculate_sleep_metrics(hypnogram):
    try: 
        # Identify Sleep Onset (SO) as the first 10 consecutive minutes of sleep
        sleep_indices = hypnogram[hypnogram['stage'] > 0].index
        SO = np.nan
        WU = np.nan

        for i in range(len(sleep_indices) - 20):
            if all(hypnogram.loc[sleep_indices[i:sleep_indices[i] + 20], 'stage'] > 0):
                SO = sleep_indices[i] 
                break

        # Identify Wake Up (WU) as the last 10 consecutive minutes of sleep
        for i in range(len(sleep_indices) - 20, 0, -1):
            if all(hypnogram.loc[sleep_indices[i:sleep_indices[i] + 20], 'stage'] > 0):
                WU = (sleep_indices[i + 19] + 1)  # To ensure we cover 10 minutes
                break

        # Calculate WASO
        if not np.isnan(SO) and not np.isnan(WU):
            sleep_period = hypnogram.loc[int(SO):int(WU)]
            WASO = len(sleep_period[sleep_period['stage'] == 0]) 
        else:
            WASO = np.nan

        # Check for long WASO periods that might be wake periods
        if not np.isnan(SO) and not np.isnan(WU):
            long_waso_periods = sleep_period['stage'].rolling(window=240).sum() == 0
            if any(long_waso_periods):
                WU = long_waso_periods.idxmax()

        # Calculate TST
        if not np.isnan(SO) and not np.isnan(WU) and (WU - SO - WASO) >= 180:
            TST = WU - SO - WASO
        else:
            TST = np.nan


        return SO, WU, TST, WASO
    
    except Exception as e:
        print(e)
        return np.nan, np.nan, np.nan, np.nan

def summary_data_dreem(subject_id, tz_str, data_path):
    try: 

        algo_dir = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/{subject_id}/fft/'
        yasa_dir = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/{subject_id}/yasa/'

        algo_files = [file for file in sorted(os.listdir(algo_dir)) if file.startswith('so_')]
        yasa_files = [file for file in sorted(os.listdir(yasa_dir)) if file.startswith('yasa_')]

        df_sleep_all = pd.DataFrame()

        input_folder = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/{subject_id}/dreem_reports/'


        for filename in sorted(os.listdir(input_folder)):
            if filename.endswith('endpoints.csv'):

                df_sleep = pd.read_csv(f'{input_folder}{filename}')

                df_pivot = df_sleep.pivot(index=['STUDYID', 'SUBJID', 'REC_DATE_TIME', 'OFFHEAD'], columns='ENDPOINT', values=['VALUE', 'QI_INDEX'])

                df_pivot.columns = [f'{level2}_{level1}' if level1 == 'QI_INDEX' else f'{level2}' for level1, level2 in df_pivot.columns]

                # Reset index to turn the previously indexed columns back into regular columns
                df_sleep = df_pivot.reset_index()

                df_sleep_all = pd.concat([df_sleep_all,df_sleep])

        df_sleep_all['start_rec'] = pd.to_datetime(df_sleep_all['REC_DATE_TIME'], utc=True)

        target_timezone = pytz.timezone(tz_str) 

        df_sleep_all['start_rec'] = df_sleep_all['start_rec'].dt.tz_convert(target_timezone)

        df_sleep_all.drop('REC_DATE_TIME', axis=1,inplace= True)

        df_sleep_all['TIB'] = df_sleep_all['TRT']/60
        df_sleep_all = df_sleep_all[df_sleep_all['TST'] >=180]
        df_sleep_all['SO'] = df_sleep_all['start_rec'] + pd.to_timedelta(df_sleep_all['SOL'], unit='m')
        df_sleep_all['FA'] = df_sleep_all['start_rec'] + pd.to_timedelta(df_sleep_all['TST'], unit='m') + pd.to_timedelta(df_sleep_all['WASO'], unit='m')
        df_sleep_all['date'] = pd.to_datetime(df_sleep_all['FA_dreem']).dt.date

        reorder = ['TIB', 'TST', 'WASO','SO','FA'] + [col for col in df_sleep_all.columns if col not in ['TIB', 'TST', 'WASO','SO','FA']]


        df_sleep_all = df_sleep_all[reorder]

        df_sleep_all['subject'] = subject_id
        #code for creating df for sleep algo results

        date = []
        so_index = []
        fa_index = []


        for file in algo_files:
            date.append(file.split('_')[2][:10])
            so_index.append(np.load(os.path.join(algo_dir, file))[0])
            fa_index.append(np.load(os.path.join(algo_dir, file))[1])

        results = {
            'date': date,
            'so_index': so_index,
            'fa_index': fa_index
        }

        # Construct the DataFrame
        df = pd.DataFrame(results)

        df['date'] = pd.to_datetime(df['date']).dt.date

        # Merge the two DataFrames on 'date'

        df_sleep_temp = pd.merge(df, df_sleep_all, on='date')

        df_sleep_temp['SO_algo'] = df_sleep_temp['start_rec'] + pd.to_timedelta(df_sleep_temp['so_index']//2, unit='m')
        df_sleep_temp['WU_algo'] = df_sleep_temp['start_rec'] + pd.to_timedelta(df_sleep_temp['fa_index']//2, unit='m')

                # df.to_csv(os.path.join(algo_dir,f'algo_SoFa_{subject_id}.csv'))

        df_sleep_yasa = pd.DataFrame()

        for filename in yasa_files:

            if filename.startswith(f'yasa_{subject_id}'):
                date = filename.split('_')[2][:-4]
                df_yasa = pd.read_csv(f'{yasa_dir}{filename}')

                df_yasa.rename(columns={'Highest_Conf_Score_Val': 'stage'}, inplace=True)

                # Get the start recording time from the date
                # start_rec = pd.to_datetime(date, utc=True).tz_convert(pytz.timezone(tz_str))

                # Calculate sleep metrics
                SO, WU, TST, WASO = calculate_sleep_metrics(df_yasa)

                # Append results
                new_row = pd.DataFrame([{
                    'subject': subject_id,
                    'date': date,
                    'so_index': SO,
                    'fa_index': WU,
                    'TST': TST,
                    'WASO': WASO
                }])

                df_sleep_yasa = pd.concat([df_sleep_yasa, new_row], ignore_index=True)
        # Save the results
        df_sleep_temp['date'] = pd.to_datetime(df_sleep_temp['date']).dt.strftime('%Y-%m-%d')
        df_sleep_yasa['date'] = pd.to_datetime(df_sleep_yasa['date']).dt.strftime('%Y-%m-%d')
        sleep_measures_all = df_sleep_temp.merge(df_sleep_yasa, on=['subject', 'date'], suffixes=('', '_dreem'),how ='left')

        sleep_measures_all['SO_yasa'] = sleep_measures_all['start_rec'] + pd.to_timedelta(sleep_measures_all['so_index_dreem']//2, unit='m')
        sleep_measures_all['FA_yasa'] = sleep_measures_all['start_rec'] + pd.to_timedelta(sleep_measures_all['fa_index_dreem']//2, unit='m')
        sleep_measures_all.to_csv(f'{yasa_dir}/dreem_sleep_summary_{subject_id}.csv', index=False)
        
        return(sleep_measures_all)
    
    except Exception as e:
        print(e)
        return None
