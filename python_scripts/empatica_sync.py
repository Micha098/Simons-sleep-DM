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

# Run code to generate report update

command = [
    'python', '/mnt/home/mhacohen/python_files/empatica_raw.py',
]
subprocess.run(command)
timer.sleep(60 * 60)  # 60 minutes * 60 seconds

command = [
    'sbatch', '/mnt/home/mhacohen/slurm_files/slurm_zcy_job.sh',
]
subprocess.run(command)

# Run local version of activity count generator

import zcy_sleep_measures