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
from withings_process import process_withings_shared_data



sync_command = f"aws s3 sync s3://sf-sleep-study/ '/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/withings' --profile mhacohen --region us-east-1"

subprocess.run(sync_command, shell=True)
subprocess.run(f"{sync_command} > output.txt", shell=True)

                      
data_path= f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/withings/'
subject_ids = pd.read_csv(f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv').drop('Unnamed: 0', axis= 1)['id']
tzs_str= pd.read_csv(f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv').drop('Unnamed: 0', axis= 1)['tz_str']


for user,tz_str in zip(subject_ids,tzs_str):
    
    output_share_dir = f"/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/data_share/{user}/withings/"

    process_withings_shared_data(user,tz_str,data_path,output_share_dir)
    
    for file in os.listdir(output_share_dir):
        if file.startswith('withings'):
            os.remove(os.path.join(output_share_dir,file))
