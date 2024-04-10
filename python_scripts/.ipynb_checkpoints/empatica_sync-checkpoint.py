#!/usr/bin/env python
import os
import subprocess
import pandas as pd
import datetime as dt
from datetime import datetime
from datetime import timedelta
from datetime import time
import shutil
import json
import glob 
import pytz
import numpy as np
import pandas as pd
import re
import scipy
import seaborn as sns
from scipy import signal
import ast
import time as timer

# code for AWS data pull

sync_command = f"aws s3 sync {os.environ['ACCESS_URL']} {os.environ['LOCAL_PATH']} --region us-east-1"
subprocess.run(sync_command, shell=True)
subprocess.run(f"{sync_command} > output.txt", shell=True)


# Get necessaery subject data  
subject_ids = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['id'].tolist()
subject_tzs = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['tz'].tolist()
subject_tzs_str = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['tz_str'].tolist()

# activate slurm for agg data

command = [
    'sbatch', '/mnt/home/mhacohen/slurm_files/slurm_agg.sh',
]
subprocess.run(command)


# wait 60 minutes hour 
timer.sleep(60 * 60)  # 60 minutes * 60 seconds


# activate summrized sleep data code

command = [
    'sbatch', '/mnt/home/mhacohen/slurm_files/slurm_summ_emp.sh',
]
subprocess.run(command)

timer.sleep(30 * 60)  # 60 minutes * 60 seconds


## code for nights report

all_users = []
all_table = []

directory = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/'

for subject in [subject for subject in sorted(os.listdir(directory)) if subject.startswith('1') or subject.startswith('3')]:
    try:
        user_nights = []
        table_nights = []
        # Construct the path for the current subject's data
        path = f'{directory}{subject}/'
        
        # Change the path to your local path if you use a local version 
        for measure in ['withings_measures','empatica_measures','dreem_reports']:
                os.makedirs(path+measure, exist_ok=True)

                num_files = len([filename for filename in sorted(os.listdir(path+measure)) if filename.startswith(measure)])
                num_rows = (len(sum_table.dropna(subset=['TST','WASO'])))

                if measure == 'dreem_reports':

                    num_files = len([filename for filename in sorted(os.listdir(path+measure)) if filename.startswith('simons_sleep')])
                try:

                    sum_table = pd.read_csv(glob.glob(f'{path+measure}/*summary*', recursive=True)[0])
                    new_path = f'{directory}sumnary_tabels/{subject}/'

                    sum_table.to_csv(f'{new_path}/{measure}_{subject}.csv')
                except Exception as e:
                    print(f'{subject} {e}')


                user_nights.append(num_files)
                table_nights.append(num_rows)

        all_users.append([subject,user_nights])
        all_table.append([subject,table_nights])

    except:
        continue

night_report = [[item[0]] + item[1] for item in all_users]
tables_report = [[item[0]] + item[1] for item in all_table]

night_reports = pd.DataFrame(night_report, columns=['Subject_ID', 'Withings_nights', 'Empatica_days', 'Dreem_nights'])
tables_reports = pd.DataFrame(tables_report, columns=['Subject_ID', 'Withings_valid_nights', 'Empatica_valid_days', 'Dreem_valid_nights'])
night_reports = night_reports.merge(tables_reports, on='Subject_ID')
night_reports['user_type'] = night_reports['Subject_ID'].apply(lambda x: 'ASD' if str(x).startswith('1') else 'NASD')
night_reports.sort_values('Subject_ID', inplace=True)
night_reports.to_csv(f'{directory}/sumnary_tabels/nights_reports.csv', index=False)


command = [
    'sbatch', '/mnt/home/mhacohen/slurm_files/slurm_raw.sh',
]
subprocess.run(command)