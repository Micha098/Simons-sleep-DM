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

def dreem_allocation():
    
    unique_cases_316= ['simons_sleep17@beacon.study_2023-12-27T02:06:59',
     'simons_sleep17@beacon.study_2024-01-01T03:09:26',
     'simons_sleep17@beacon.study_2024-01-02T05:20:43',
     'simons_sleep17@beacon.study_2023-12-29T04:09:13',
     'simons_sleep17@beacon.study_2023-12-24T01:31:17',
     'simons_sleep17@beacon.study_2023-12-23T01:47:07',
     'simons_sleep17@beacon.study_2024-01-02T15:26:22',
     'simons_sleep17@beacon.study_2023-12-28T02:23:45',
     'simons_sleep17@beacon.study_2023-12-31T02:05:55',
     'simons_sleep17@beacon.study_2023-12-26T01:48:37']

    data_check = pd.read_csv('/mnt/home/mhacohen/Participants and devices - Dreem data check.csv')[['Subject RID','User ID'	,'starting date','ending date','Dreem user id']]

    data_check = data_check.dropna(subset=['starting date', 'ending date'], how='all')


    data_check.loc[data_check['ending date'].isna(), 'ending date'] = dt.date.today()
    data_check['starting date'] = pd.to_datetime(data_check['starting date'])
    data_check['ending date'] = pd.to_datetime(data_check['ending date'], errors='coerce', dayfirst=True)


    data_check.dropna(subset=['starting date', 'ending date'], inplace=True)

    # Initialize dictionary

    mapping = {}


    for index, row in data_check.iterrows():

        data_tuple = (row['User ID'], str(row['starting date']), str(row['ending date']))
        dreem_user = row['Dreem user id'].split('@')[0]

        # If the Dreem user id is already a key in the mapping, append the tuple
        if dreem_user in mapping:
            dreem_user = row['Dreem user id'].split('@')[0]
            mapping[dreem_user].append(data_tuple)
            # Correct record on wrong sibling
            if dreem_user == 'simons_sleep29':
                mapping[dreem_user].append(('U325', '2024-01-27 00:00:00', '2024-01-30 00:00:00'))
        else:
            # Otherwise, create a new list with the tuple
            mapping[dreem_user] = [data_tuple]


    def adjust_date_for_recording_time(file_date_str, file_time_str):
        file_datetime = datetime.strptime(f"{file_date_str}T{file_time_str}", "%Y-%m-%dT%H:%M:%S%z")

        # Adjusting the date if the recording started before midnight.
        if file_datetime.hour < 15:  # Assuming recordings start late in the evening or just after midnight.
            adjusted_datetime = file_datetime - timedelta(days=1)
        else:
            adjusted_datetime = file_datetime
        return adjusted_datetime.date()

    def find_participant_id(simons_sleep_id, file_date):
        for entry in mapping.get(simons_sleep_id, []):
            participant_id, start_date, end_date = entry
            start_date_parsed = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").date()
            end_date_parsed = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").date()
            if start_date_parsed <= file_date <= end_date_parsed:
                return participant_id
        return None

    def move_files(source_dir, dest_base_dir):
        for file_type,dir_type in zip(['edf/Prod','hypnogram/Prod','endpoints/Prod'],
                                      ['edf','hypno','dreem_reports']):
            file_dir = source_dir+file_type
            for filename in os.listdir(file_dir):
                if not filename.endswith(('.edf', '_hypnogram.txt', '_endpoints.csv')):

                    continue
                # Extract simons_sleep ID and recording start time from the filename.
                if filename[:47] in unique_cases_316:
                    print(f'found unique U316 file {filename}')
                    try:
                        dest_dir = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/316/{dir_type}/'

                        if filename not in os.listdir(dest_dir):
                            shutil.move(os.path.join(file_dir, filename), os.path.join(dest_dir, filename))

                    except:
                        continue
                else:

                    simons_sleep_id, timestamp = filename.split('@')[0], filename.split('_')[2]
                    file_date_str, file_time_str = timestamp.split('T')
                    file_time_str = file_time_str.split('.')[0]
                    file_date = adjust_date_for_recording_time(file_date_str, file_time_str)
                    participant_id = find_participant_id(simons_sleep_id, file_date)

                    if participant_id is not None:
                        participant_id = participant_id[1:]
                        dest_dir = f"{dest_base_dir}/{participant_id}/{dir_type}/"

                        if not os.path.exists(dest_dir):
                            os.makedirs(dest_dir)
                        if filename not in os.listdir(dest_dir):
                            shutil.copy(os.path.join(file_dir, filename), os.path.join(dest_dir, filename))

    #excute

    source_dir = "/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/dreem/aws_data/"
    dest_base_dir = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/'

    # activate data acllocation function 
    move_files(source_dir, dest_base_dir)
