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
# Import local scripts
from dreem_allocation_share import dreem_allocation
from dreem_summ_data import summary_data_dreem

# code for pulling the data from aws
os.environ['LOCAL_PATH'] = "/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/dreem/aws_data/"

# aws configure AWS Access Key ID [None]: AWS_ACCESS_KEY_ID AWS Secret Access Key [None]: AWS_SECRET_ACCESS_KEY

sync_command = f"aws s3 sync {os.environ['ACCESS_URL']} {os.environ['LOCAL_PATH']} --region us-east-1"

subprocess.run(sync_command, shell=True)

# Create list of unique cases and run dreem mapping code 

dreem_allocation()

# get sleep summary stats from dreem

# iterate across users and get summrized data


command = [
    'python', '/mnt/home/mhacohen/python_files/dreem_hypno_share.py',
]

subprocess.run(command)


# timer.sleep(10 * 60)  # 60 minutes * 60 seconds

# command = [
#     'sbatch', '/mnt/home/mhacohen/slurm_files/slurm_dreem_edf.sh',
# ]
# subprocess.run(command)

