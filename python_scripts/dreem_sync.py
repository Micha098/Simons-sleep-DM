#!/usr/bin/env python
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
from dreem_allocation import dreem_allocation
from dreem_summ_data import summary_data_dreem

# code for pulling the data from aws

# aws configure AWS Access Key ID [None]: AWS_ACCESS_KEY_ID AWS Secret Access Key [None]: AWS_SECRET_ACCESS_KEY

sync_command = f"aws s3 sync {os.environ['ACCESS_URL']} {os.environ['LOCAL_PATH']} --region us-east-1"

subprocess.run(sync_command, shell=True)


subject_ids = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['id'].tolist()
subject_tzs = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['tz'].tolist()
subject_tzs_str = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['tz_str'].tolist()

output_dir = "/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/"
summary_dir = "dreem_reports/"


# Create list of unique cases

dreem_allocation()

# get sleep summary stats from dreem


# iterate across users and get summrized data


for participant_id,tz_str in zip(subject_ids,subject_tzs_str):
    
    try:

        path = os.path.join(f'{output_dir}{participant_id}', summary_dir)
        
        os.makedirs(path, exist_ok=True)

        # Generate the filename
        filename = f"dreem_nights_summary_{participant_id}.csv"

        filepath = os.path.join(path, filename)

        df_dreem_rep = summary_data_dreem(participant_id,tz_str,path)
        df_dreem_rep[['date','start_rec','TIB','TST','WASO','SO','FA','SO_algo','WU_algo']].to_csv(filepath)
    
    except Exception as e:
        print(f"Error processing data for subject_id {participant_id}: {e}")
        continue


command = [
    'sbatch', '/mnt/home/mhacohen/slurm_files/slurm_dreem_hypno.sh',
]

subprocess.run(command)

command = [
    'sbatch', '/mnt/home/mhacohen/slurm_files/slurm_dreem_job.sh',
]
subprocess.run(command)
# wait 60 minutes hour 

