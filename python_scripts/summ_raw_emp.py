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
subject_ids = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['id'].drop_duplicates().tolist()
subject_tzs = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['tz'].tolist()
subject_tzs_str = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['tz_str'].tolist()

user = subject_ids[i]
tz_str = subject_tzs_str[i]
zcy_all = pd.DataFrame()

input_dir = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/{user}/zcy/'
output_dir = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/{user}/sadeh/'
agg_dir = f'/mnt/home/mhacohen/ceph/agg_data/agg_emp_all_{user}.csv'

for filename in sorted(os.listdir(input_dir)):
    if filename.startswith(f'empatica_zcy_{user}') and 'sadeh' in pd.read_csv(f'{input_dir}/{filename}').columns:
        dfi = pd.read_csv(f'{input_dir}/{filename}')
        zcy_all = pd.concat([dfi,zcy_all])

if 'Unnamed: 0' in zcy_all.columns:
    zcy_all.drop('Unnamed: 0',inplace= True,axis=1)

zcy_all.sort_values('date', inplace = True)  
zcy_all.reset_index(drop = True, inplace = True) 

agg_data = pd.read_csv(f'{agg_dir}').rename(columns = {'timestamp_iso':'date'})

zcy_all = zcy_all.merge(agg_data[['date','wearing_detection_percentage','sleep_detection_stage']], how='left', on = 'date') 

zcy_all.sort_values('date', inplace = True)
zcy_all.reset_index(drop = True, inplace = True)
zcy_all.loc[(zcy_all['sadeh'].isna()), ['sadeh']] = 0

zcy_all.loc[(zcy_all['wearing_detection_percentage'] <= 50) | (zcy_all['wearing_detection_percentage'].isna()), ['sadeh', 'ZCY']] = np.nan

target_timezone = pytz.timezone(tz_str) 

zcy_all['date'] = pd.to_datetime(zcy_all['date'], errors='coerce', utc=True)
zcy_all['date'] = zcy_all['date'].dt.tz_convert(target_timezone)

zcy_all['date'] = pd.to_datetime(zcy_all['date'])
zcy_all.set_index('date', inplace=True)

# Resample the data to ensure every minute is included
zcy_all = zcy_all[~zcy_all.index.duplicated(keep='first')]

zcy_all.resample('T').asfreq()


zcy_all.sort_values('date', inplace = True)  

zcy_all.reset_index(inplace = True)
zcy_all.drop_duplicates(inplace=True) 

zcy_all.reset_index(drop = True, inplace = True)

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
                print(df['date'][fa_index])
                print(fa_index)

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

    # os.makedirs(output_dir, exist_ok=True)

    sleep_measures_per_night = sleep_measures_per_night[['date','TST','WASO','FA','SO']]#dfd.columns]

    os.makedirs(output_dir, exist_ok=True)


sleep_measures_per_night.to_csv(f'{output_dir}empatica_raw_nights_{user}.csv')
print(f'completed user {user}')