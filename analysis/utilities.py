import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import core
import os
import re

# Utility file for project settings

project_data_dir = {'Dreem':'/mnt/ceph/users/mhacohen/data/Dreem',
                   'NSRR': '/mnt/home/geylon/ceph/data/NSRR/nchsdb/sleep_data',
                   'NSRR_Stages':'/mnt/ceph/users/mhacohen/data//NSRR/stages'}
epoch_data_dir ={'NSRR': '/mnt/home/mhacohen/ceph/data/processed_data/nsrr/resampled_data/nchsdb' }

# Function to find the last processed file in the directory
def find_last_processed_file(directory, files_to_process_df):
    # List all files in the directory
    files = os.listdir(directory)

    # Filter files matching the pattern
    pattern = r'^predict_usleep_(\d+)-128Hz-raw\.csv$'
    matching_files = [file for file in files if re.match(pattern, file)]

    # Sort the matching files by last modified timestamp in descending order
    matching_files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)

    # Find the first matching file that is also in the DataFrame
    for file in matching_files:
        if file in files_to_process_df['File-Name'].values:
            return file

    return None


                   

                   