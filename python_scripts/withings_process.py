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


def process_withings_shared_data(subject_id,tz,data_path,output_path):
        try:
            input_folder_wit = glob.glob(f'{data_path}/**/*{subject_id}*/', recursive=True)[0]
        except Exception as e:
            print(f"Error processing data for subject_id {subject_id} {e}")
            return None
        data_folders = sorted([folder for folder in os.listdir(input_folder_wit) if folder.isdigit()], key=int, reverse=True)


        input_zip = None

        # Iterate over the folders from newest to oldest
        for data_folder in data_folders:
            input_folder_wit_current = os.path.join(input_folder_wit, data_folder)
            potential_zip_files = glob.glob(f'{input_folder_wit_current}/*data*.zip', recursive=True)

            for zip_path in potential_zip_files:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    if any("raw_bed_Pressure.csv" in file for file in zip_ref.namelist()):
                        input_zip = zip_path  # Take the first .zip file found

            break
                        
        
        if input_zip:
            # Open the zip file
            with zipfile.ZipFile(input_zip, 'r') as zip_ref:
                # Extract all the contents into the directory
                zip_ref.extractall(input_folder_wit)

                for file in os.listdir(input_folder_wit):
                    if file in []
                    
                        if file.endswith('.csv'):
                            dfi = pd.read_csv(os.path.join(input_folder_wit,file))
                            if dfi.empty:
                                continue
                            if 'start' in dfi.columns:
                                date_columns = 'start'

                            elif 'date' in dfi.columns:
                                date_columns = 'date'

                            elif 'Date' in dfi.columns:
                                date_columns = 'Date'

                            elif 'from' in dfi.columns:
                                date_columns = ['from','to']

                                dfi['from'] = pd.to_datetime(dfi['from'],utc=True)
                                dfi['to'] = pd.to_datetime(dfi['to'],utc=True)

                                target_timezone = pytz.timezone(tz_str) 

                                dfi['from'] = dfi['from'].dt.tz_convert(target_timezone)
                                dfi['to'] = dfi['to'].dt.tz_convert(target_timezone)

                                dfi.to_csv(os.path.join(output_path,file), index = False)

                                continue


                            else:
                                dfi.to_csv(os.path.join(output_path,file), index = False)

                                continue
                            
                        dfi.sort_values(date_columns, inplace = True)

                        dfi[date_columns] = pd.to_datetime(dfi[date_columns],utc=True)

                        dfi.drop_duplicates(inplace=True)

                        target_timezone = pytz.timezone(tz_str) 

                        dfi[date_columns] = dfi[date_columns].dt.tz_convert(target_timezone)

                        dfi.to_csv(os.path.join(output_path,file), index = False)
                        
data_path= f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/withings/'
subject_ids = pd.read_csv(f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv').drop('Unnamed: 0', axis= 1)['id']
tzs_str= pd.read_csv(f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv').drop('Unnamed: 0', axis= 1)['tz_str']
