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
tz_str = tzs_str[i]
results = []
# Intialize lists
TST = []
WASO = [] 
SO = []
FA = []

waso_segment_length = 120 # minutes


directory =f"/mnt/home/mhacohen/ceph/agg_data/"
output_dir = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/{user}/empatica_measures/'

for filename in sorted(os.listdir(directory)):
    if filename.startswith(f'agg_emp_all_{user}'):

        df = pd.read_csv(f'{directory}/{filename}')

        target_timezone = pytz.timezone(tz_str) 

        # Apply the conversion function to the column
        df['timestamp_iso'] = pd.to_datetime(df['timestamp_iso'], errors='coerce', utc=True)
        df['timestamp_iso'] = df['timestamp_iso'].dt.tz_convert(target_timezone)

        df.drop_duplicates('timestamp_iso', keep = 'first')

        df.reset_index(inplace = True)

        # Correct for Waso_index shoerter then 3 minutes
        df['sleep_stage_fixed'] = df['sleep_detection_stage']
        df.drop('timestamp_unix', axis=1, inplace=True)

        for i in range(1, len(df) - 1):
            if df.iloc[i]['sleep_detection_stage'] == 102:
                if (df.iloc[i - 1: i + 2]['sleep_detection_stage'] == 101).sum() >= 1:
                    df.at[df.index[i], 'sleep_stage_fixed'] = 101
        print('sleep_stage_fixed')

        df.drop('index', axis=1, inplace=True)
        df.drop_duplicates(inplace= True)
        df.reset_index(drop = True, inplace=True)
        results = []

        # Initialize the first search index
        start_index = 0

        while start_index < len(df) - 10:
            so_index = None
            fa_index = None

            # Search for sleep onset (so)
            for i in range(start_index, len(df)-10):
                if all(df.iloc[i:i+10]['sleep_stage_fixed'] == 101):
                    so_index = i
                    break

            if so_index is not None:
                # Look for FA starting from the end of the detected SO.
                # Find 2 hours of consecutive wakefullness
                for i in range(so_index + 10, so_index+(60*14)):
                    if all(df.iloc[i:i+120]['sleep_stage_fixed'].isin([102, 300, 0,np.nan])):
                        print(f'{i}-fa')
                        # Iterate backwards to find the last 10 consecutive sleep epochs
                        for j in range(i - 1, so_index + 9, -1):
                            if all(df.iloc[j-9:j+1]['sleep_stage_fixed'] == 101):
                                fa_index = j
                                break
                        if fa_index is None:
                            fa_index = i
                        break


                if fa_index is not None:
                    tst = (df.loc[so_index:fa_index, 'sleep_stage_fixed'] == 101).sum()
                    waso = df.loc[so_index:fa_index, 'sleep_stage_fixed'].isin([102, 300, 0,np.nan]).sum()

                    if (fa_index - so_index - waso) > 180: # check for short or fregmented sleep periods
                        SO.append(df['timestamp_iso'].iloc[so_index])
                        FA.append(df['timestamp_iso'].iloc[fa_index])
                        TST.append(tst)
                        WASO.append(waso)
                        results.append({'TST': tst, 'WASO': waso, 'SO': SO[-1], 'FA': FA[-1]})
                        print(f'TST: {tst}, WASO: {waso} SO: {SO[-1]}, FA: {FA[-1]}')
                    # Update start_index to the index right after the current FA for the next cycle
                    start_index = fa_index + 1
                else:
                    # If no FA found, continue searching from 12h later
                    start_index = start_index + 60*12
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

    os.makedirs(output_dir, exist_ok=True)


sleep_measures_per_night.to_csv(f'{output_dir}empatica_nights_summary_{user}.csv')
print(f'completed user {user}')


directory =f"/mnt/home/mhacohen/ceph/agg_data/"
output_dir = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/{user}/empatica_measures/'

for filename in sorted(os.listdir(directory)):
    if filename.startswith(f'agg_emp_all_{user}'):

        df = pd.read_csv(f'{directory}/{filename}')

        target_timezone = pytz.timezone(tz_str) 

        # Apply the conversion function to the column
        df['timestamp_iso'] = pd.to_datetime(df['timestamp_iso'], errors='coerce', utc=True)
        df['timestamp_iso'] = df['timestamp_iso'].dt.tz_convert(target_timezone)

        df.drop_duplicates('timestamp_iso', keep = 'first')

        df.reset_index(inplace = True)

        # Correct for Waso_index shoerter then 3 minutes
        df['sleep_stage_fixed'] = df['sleep_detection_stage']
        df.drop('timestamp_unix', axis=1, inplace=True)

        for i in range(1, len(df) - 1):
            if df.iloc[i]['sleep_detection_stage'] == 102:
                if (df.iloc[i - 1: i + 2]['sleep_detection_stage'] == 101).sum() >= 1:
                    df.at[df.index[i], 'sleep_stage_fixed'] = 101
        print('sleep_stage_fixed')

        df.drop('index', axis=1, inplace=True)
        df.drop_duplicates(inplace= True)
        df.reset_index(drop = True, inplace=True)
        results = []

        # Initialize the first search index
        start_index = 0

        while start_index < len(df) - 10:
            so_index = None
            fa_index = None

            # Search for sleep onset (so)
            for i in range(start_index, len(df)-10):
                if all(df.iloc[i:i+10]['sleep_stage_fixed'] == 101):
                    so_index = i
                    break

            if so_index is not None:
                # Look for FA starting from the end of the detected SO.
                # Find 2 hours of consecutive wakefullness
                for i in range(so_index + 10, so_index+(60*14)):
                    if all(df.iloc[i:i+120]['sleep_stage_fixed'].isin([102, 300, 0,np.nan])):
                        print(f'{i}-fa')
                        # Iterate backwards to find the last 10 consecutive sleep epochs
                        for j in range(i - 1, so_index + 9, -1):
                            if all(df.iloc[j-9:j+1]['sleep_stage_fixed'] == 101):
                                fa_index = j
                                break
                        if fa_index is None:
                            fa_index = i
                        break


                if fa_index is not None:
                    tst = (df.loc[so_index:fa_index, 'sleep_stage_fixed'] == 101).sum()
                    waso = df.loc[so_index:fa_index, 'sleep_stage_fixed'].isin([102, 300, 0,np.nan]).sum()

                    if (fa_index - so_index - waso) > 180: # check for short or fregmented sleep periods
                        SO.append(df['timestamp_iso'].iloc[so_index])
                        FA.append(df['timestamp_iso'].iloc[fa_index])
                        TST.append(tst)
                        WASO.append(waso)
                        results.append({'TST': tst, 'WASO': waso, 'SO': SO[-1], 'FA': FA[-1]})
                        print(f'TST: {tst}, WASO: {waso} SO: {SO[-1]}, FA: {FA[-1]}')
                    # Update start_index to the index right after the current FA for the next cycle
                    start_index = fa_index + 1
                else:
                    # If no FA found, continue searching from 12h later
                    start_index = start_index + 60*12
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

    os.makedirs(output_dir, exist_ok=True)


sleep_measures_per_night.to_csv(f'{output_dir}empatica_nights_summary_{user}.csv')
print(f'completed user {user}')