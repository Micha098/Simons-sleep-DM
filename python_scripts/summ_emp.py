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


i = int(sys.argv[1])

# initialize variables 
agg_emp_all = pd.DataFrame()
df_temp = pd.DataFrame()
aggregated_data_path = None
    
users = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['id'].tolist()
tzs = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['tz'].tolist()
tzs_str = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['tz_str'].tolist()

user = users[i]

results = []


directory =f"/mnt/home/mhacohen/ceph/agg_data/"
output_dir = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/{user}/empatica_measures/'

nights_summary_emp = pd.DataFrame()   

for filename in sorted(os.listdir(directory)):
    if filename.startswith(f'agg_emp_all_{user}'):

        df = pd.read_csv(f'{directory}/{filename}')
        df.timestamp_unix = pd.to_datetime(df.timestamp_unix)
        df.drop_duplicates('timestamp_unix', keep = 'first')

        df.reset_index(inplace = True)

        df['timestamp_iso'] = pd.to_datetime(df['timestamp_iso'])

        # Correct for Waso shoerter then 3 minutes
        df['sleep_stage_fixed'] = df['sleep_detection_stage']

        for i in range(1, len(df) - 1):
            if df.iloc[i]['sleep_detection_stage'] == 102:
                if (df.iloc[i - 1: i + 2]['sleep_detection_stage'] == 101).sum() >= 1:
                    df.at[df.index[i], 'sleep_stage_fixed'] = 101

        df['ref_time'] = df['timestamp_iso'] - timedelta(hours = 10)
        df['ref_time'] = pd.to_datetime(df['ref_time'], errors='coerce')

        for date, group_df in df.groupby(df['ref_time'].dt.date):

            try:
                group_df.reset_index(inplace=True)

                group_df['timestamp_iso'] = pd.to_datetime(group_df['timestamp_iso'], errors='coerce')

                # TST: Total minutes of sleep_stage_fixed == 1 per night
                WASO = group_df['sleep_stage_fixed'].isin([102, 300])
                # WASO: Total minutes awake after sleep has been initiated per night

                TST = group_df['sleep_stage_fixed'] == 101 # Replace with actual WASO calculation

                # Secnondary WASO and TST calculation
                fa_index = None
                waso_segment_length = 120

                for i in range(len(group_df) - waso_segment_length):
                    if all(group_df.iloc[i:i + waso_segment_length]['sleep_stage_fixed'].isin([102, 300])):
                        # Find the last sleep timepoint before this WASO segment
                        last_sleep_before_waso = group_df.iloc[:i][group_df['sleep_stage_fixed'] == 101].iloc[-1]
                        fa_index = last_sleep_before_waso.name
                        print('found 120 min waso')
                        break


                if fa_index:
                # Set FA to the last sleep timepoints before the 120 consecutive 101 values
                    fa = group_df['timestamp_iso'].iloc[fa_index]

                    # Recalculate TST: Total sleep time from the beginning of the recording to the new FA
                    TST = df.loc[group_df.index <= fa_index, 'sleep_stage_fixed'].isin([102, 300]).sum()

                else:

                    # Determine the last index before 14:00:00 in the current group
                    last_before_14 = group_df[group_df['timestamp_iso'].dt.time < pd.to_datetime('14:00:00').time()].last_valid_index()

                    # Initialize fa_index
                    fa_index = None

                    # Adjusted FA search starting from the last index before 14:00:00
                    if last_before_14 is not None:  # Make sure there is at least one record before 14:00:00
                        for i in range(last_before_14, 9, -1):  # Adjusted range to safely use i:i+10 below
                            if all(group_df.iloc[i-9:i+1]['sleep_stage_fixed'] == 101):  # Adjusted to check 10 values including i
                                fa_index = i  # Use the last index of the 10 consecutive '101' values
                                break

                    if fa_index is None:  # If no segment of 10 consecutive 101s, find the last 101 before 14:00:00
                        fa_condition = (group_df['sleep_stage_fixed'] == 101) & (group_df['timestamp_iso'].dt.time < pd.to_datetime('14:00:00').time())
                        fa_candidate = group_df[fa_condition].last_valid_index()
                        fa = group_df.loc[fa_candidate, 'timestamp_iso'] if fa_candidate is not None else None
                    else:
                        fa = group_df.iloc[fa_index]['timestamp_iso']

                first_after_18 = group_df[group_df['timestamp_iso'].dt.time > pd.to_datetime('18:00:00').time()].first_valid_index()

                so_index = None
                for i in range(first_after_18,len(group_df) - 9):
                    if all(group_df.iloc[i:i + 10]['sleep_stage_fixed'] == 101):
                        so_index = i
                        break

                if so_index is None:  # If no segment of 10 consecutive 101s, find the first 101
                    so_condition = (group_df['sleep_stage_fixed'] == 101) & (group_df['timestamp_iso'].dt.time > pd.to_datetime('18:00:00').time())
                    so_candidate = group_df[so_condition].first_valid_index()
                    so = group_df.loc[so_candidate, 'timestamp_iso'] if so_candidate is not None else None
                else:
                    so = group_df.iloc[so_index]['timestamp_iso']

                results.append([date+timedelta(days=1),so,fa,TST.sum(),WASO.sum()])
            except Exception as e:
                print(f"Error processing data for subject_id {user}: {e}")
                continue

sleep_measures_per_night = pd.DataFrame(results, columns = ['date','so','wu','TST','WASO'])
sleep_measures_per_night.loc[sleep_measures_per_night['TST'] < 180, ['WASO', 'TST']] = None
sleep_measures_per_night.loc[sleep_measures_per_night['TST'] > 780, ['WASO', 'TST']] = None
try:
    sleep_measures_per_night.loc[np.isnat(sleep_measures_per_night['so']), ['WASO', 'TST']] = None
except:
    print('so not it sleep_measures_per_night columns' )
if 'ref_time' in sleep_measures_per_night.columns:

    sleep_measures_per_night.drop('ref_time', axis=1, inplace=True)
try:
    os.makedirs(output_dir, exist_ok=True)
    os.remove(f'{output_dir}/summary_sleep_emp_{user}.csv')
except:
    print('no file to remove')
sleep_measures_per_night.to_csv(f'{output_dir}empatica_nights_summary_101{user}.csv')
print(f'completed user {user}')
