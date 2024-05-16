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


source_dir = "/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/dreem/aws_data/"
dest_base_dir = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/'

# First extract error in filenames and repalce with correct filenames in the aws dir

unique_cases = pd.read_csv('/mnt/home/mhacohen/Participants and devices - Dreem corrections.csv')

user_wrong = unique_cases['Original Dreem Username']
user_fixed = unique_cases['Correct Dreem Username']
unique_cases['corrected_names'] = 0

for i in range(len(unique_cases['Recorded Dreem Filename'])):
    
    part1 = unique_cases['Recorded Dreem Filename'][i].split('T')[0]
    part2 = unique_cases['Recorded Dreem Filename'][i].split('T')[1].replace('-',':')

    unique_cases['Recorded Dreem Filename'][i] = part1+'T'+part2[:8]
    unique_cases['corrected_names'][i] = (unique_cases['Recorded Dreem Filename'][i].replace(user_wrong[i],user_fixed[i]))

for dir_type in ['edf/Prod','hypnogram/Prod','endpoints/Prod','h5/Prod']:
    
    for filename in os.listdir(os.path.join(source_dir,dir_type)):
        if filename.startswith('simons_sleep') and filename[:47]:
            matches = unique_cases['Recorded Dreem Filename'] == filename[:47]
            if matches.any():
                i = unique_cases.index[matches].tolist()

                old_name = os.path.join(os.path.join(source_dir,dir_type), filename)
                new_path = os.path.join(os.path.join(source_dir,dir_type), filename.replace(user_wrong[i].iloc[0],user_fixed[i].iloc[0]))
                os.rename(old_name,new_path)
                
def dreem_allocation():

    data_check = pd.read_csv('/mnt/home/mhacohen/Participants and devices - Dreem data check.csv')[['User ID','starting date','ending date','Dreem user id']]


    data_check = data_check.dropna(subset=['starting date', 'ending date'], how='all')


    data_check.loc[data_check['ending date'].isna(), 'ending date'] = dt.date.today()
    data_check['starting date'] = pd.to_datetime(data_check['starting date'])
    data_check['ending date'] = pd.to_datetime(data_check['ending date'], errors='coerce')


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

