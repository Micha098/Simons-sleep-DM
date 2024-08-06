import os
import numpy as np  # useful for many scientific computing in Python
import os
import sys
import glob
import csv
import re
import fnmatch
from math import *
import pandas as pd # primary data structure library
import scipy
from scipy.stats import kendalltau, pearsonr, spearmanr,linregress
from scipy import constants
from scipy import stats
from scipy.stats import zscore
import seaborn as sns
from scipy import constants
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import datetime as dt
from datetime import datetime
#from datetime import datetime as dt
from datetime import timedelta, timezone
from datetime import time
from utilities import project_data_dir
from eeg_sleep import EEGSleep
import argparse
from dateutil import parser as dt_parser  # Aliased to avoid conflicts
import mne
import pytz
import shutil

j = int(sys.argv[1])

subject_id = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['id'].tolist()
tzs_str = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['tz_str'].tolist()

input_folder = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/{subject_id[j]}/'
output_folder = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/data_share/{subject_id[j]}/dreem/'

date_list = [re.search(r'\d{4}-\d{2}-\d{2}', file).group() for file in os.listdir(output_folder) if file.endswith(".edf") and not file.startswith("eeg") ]
    
parser=argparse.ArgumentParser()
parser.add_argument('--project_name', type=str)
parser.add_argument('--model', type=str)
args = parser.parse_args(args=['--project_name', 'Dreem', '--model', 'YASA'])


# Set directory path to where the EDF files are located
edf_in = input_folder + '/edf/'
edf_out = output_folder + '/edf/'
csv_dir = output_folder + '/hypno/'

# Check and create directories if they don't exist
for directory in [edf_in,edf_out]:
    if not os.path.isdir(directory):
        os.makedirs(directory)
        
# # Get list of EDF files in the directory
path_edfs = sorted([file for file in os.listdir(edf_in) if file.endswith(".edf") and not file.startswith("eeg") ],reverse=True)
path_edf = None  # Initialize path_edf outside the loop

for pathi in path_edfs:
    try:
        match = re.search(r'(\d{4}-\d{2}-\d{2})T', pathi)
        if match:
            file_date = re.search(r'\d{4}-\d{2}-\d{2}', pathi).group()
            if not file_date:
                file_date = re.search(r'\d{4}:\d{2}:\d{2}', pathi).group()

    except Exception as e:
        print(f"Error processing data for subject_id {pathi}: {e}")
        continue


    path_edf = pathi
    file_date = dt.datetime.strptime(file_date, '%Y-%m-%d').date()

    if path_edf:
        try:
            
             filename = path_edf
            
            datetime_str = filename.split('@')[1].split('.')[1]
            print(datetime_str)
            datetime_str= datetime_str.split('_')[1]
            print(datetime_str)


            # Parse the datetime string into a datetime object, including the timezone
            datetime_obj = dt_parser.parse(datetime_str)
            # Define the target timezone
            target_tz = pytz.timezone(tzs_str[j])
            
            utc_falg = False
        
           if datetime_obj.tzinfo != target_tz:
                datetime_obj = datetime_obj.astimezone(target_tz)
                utc_flag = True
                            
            file_time = datetime_obj.time()
            file_tz = datetime_obj.tzinfo
            
             # Check if the time adjustment crosses over to the next day
            if file_time > dt.datetime.strptime('19:00:00', '%H:%M:%S').time():
                datetime_obj += timedelta(days=1)  # Add one day

            file_date = datetime_obj.date()

            path_edf = os.path.join(edf_in, path_edf)
            # Check the size of the file
            file_size = os.path.getsize(path_edf)  # Get file size in bytes
            size_limit = 10 * 1024 * 1024  # 10MB in bytes ~ 1 hour of recording

            if file_size < size_limit:
                print(f"File {path_edf} is smaller than 20MB and will be deleted.")
                os.remove(path_edf)  # Delete the file
                continue
            else:
                print(f'processing {path_edf}')
                # Proceed with loading the EDF file as it meets the size criteria
                EEG = mne.io.read_raw_edf(path_edf, preload=True)

            if utc_flag == True:
                datetime_obj_utc = datetime_obj.astimezone(pytz.UTC)
                datetime_obj_utc = datetime_obj_utc.replace(tzinfo=timezone.utc)
                print(f"Time adjusted to: {datetime_obj_utc}")

                EEG.set_meas_date(datetime_obj_utc)
                # issue with expotred data - Neelay:
                EEG.export(f'{path_edf}', fmt='edf', overwrite=True)
                
            
             # Check if the time adjustment crosses over to the next day

            # save new edf filename 
            new_edf_filename = f"eeg_{subject_id[j]}_{file_date}.edf"

            #update harmonized data
            new_edf_path = os.path.join(edf_out, new_edf_filename)

            print(f"EDF file {new_edf_path} was recorded on {file_date.strftime('%Y-%m-%d')}")

            # Save EEG data in EDF format
            os.rename(path_edf, new_edf_path)
        
            print(f'saved files {subject_id[j]}_{file_date}')
        except Exception as e:
            print(f"Error processing data for subject_id {pathi}: {e}")

            continue


    else:
        print(f'no files from {file_date} for {subject_id[j]}')


