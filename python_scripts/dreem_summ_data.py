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

def summary_data_dreem(subject_id,tz_str,data_path):
    
    df_sleep_all = pd.DataFrame()

    input_folder = data_path

    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith('endpoints.csv'):

            df_sleep = pd.read_csv(f'{input_folder}{filename}')

            df_pivot = df_sleep.pivot(index=['STUDYID', 'SUBJID', 'REC_DATE_TIME', 'OFFHEAD'], columns='ENDPOINT', values=['VALUE', 'QI_INDEX'])

            df_pivot.columns = [f'{level2}_{level1}' if level1 == 'QI_INDEX' else f'{level2}' for level1, level2 in df_pivot.columns]

            # Reset index to turn the previously indexed columns back into regular columns
            df_sleep = df_pivot.reset_index()

            df_sleep_all = pd.concat([df_sleep_all,df_sleep])

    df_sleep_all['start_rec'] = pd.to_datetime(df_sleep_all['REC_DATE_TIME'], utc=True)
    df_sleep_all.drop('REC_DATE_TIME', axis=1,inplace= True)
    target_timezone = pytz.timezone(tz_str) 
    df_sleep_all['start_rec'] = df_sleep_all['start_rec'].apply(lambda x: x.tz_convert(target_timezone))


    df_sleep_all['TIB'] = df_sleep_all['TRT']/60
    df_sleep_all = df_sleep_all[df_sleep_all['TST'] >=180]
    df_sleep_all['SO'] = df_sleep_all['start_rec'] + pd.to_timedelta(df_sleep_all['SOL'], unit='m')
    df_sleep_all['FA'] = df_sleep_all['start_rec'] + pd.to_timedelta(df_sleep_all['TST'], unit='m') + pd.to_timedelta(df_sleep_all['WASO'], unit='m')
    df_sleep_all['date'] = pd.to_datetime(df_sleep_all['start_rec']).dt.date

    reorder = ['TIB','TST','WASO','SO','FA'] + [col for col in df_sleep_all.columns if col not in ['TIB', 'TST', 'WASO','SO','FA']]
    df_sleep_all = df_sleep_all[reorder]

    return df_sleep_all