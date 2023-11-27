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
from datetime import timedelta
from datetime import time

from utilities import project_data_dir
from eeg_sleep import EEGSleep
import argparse

j = int(sys.argv[1]) - 1 
target_date = sys.argv[2]

subject_id = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['id'].tolist()

# Convert Dreem Hypmnogram to desired format 

directory = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/dreem/{subject_id[j]}/txt/'

if os.path.isdir(directory):
   
    output_folder = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/dreem/{subject_id[j]}/txt/csv/'

    #check folder
    os.chdir(directory)
    # Get Dreem data in the automatic txt foramt- Change file and directory name according to user Ilan/Micha
    date_list=[]

    for filename in os.listdir(directory):
            if filename.endswith("hypnogram.txt"):
                with open(f"{directory}/{filename}") as file:
                    lines = file.readlines()
                    # Find the line with "idorer Time" and extract the date
                    for i, line in enumerate(lines):
                        if "Scorer Time" in line:
                            date = re.search(r"\d{2}/\d{2}/\d{2}", line).group()
                            time = re.search(r"\d{2}:\d{2}:\d{2}", line).group()

                            date = dt.datetime.strptime(date, '%m/%d/%y')
                            time = dt.datetime.strptime(time, '%H:%M:%S').time()
                            if time > dt.time(19, 0, 0):
                                date += timedelta(days=1)

                            date = date.strftime("%Y-%m-%d")
                            date_list.append(date)
                            new_filename = f"dreem_{subject_id[j]}_{date}.txt"
                            # Delete all lines before the table
                            lines = lines[(i+2):]
                            break
                    # Write the remaining lines to a new file with the new name
                    with open(f"{directory}/{new_filename}", "w") as new_file:
                        new_file.writelines(lines)
                    locals()[f'dreem_{subject_id[j]}_{date}'] = pd.read_csv(f"{directory}/{new_filename}",sep= "\t")

                    df = locals()[f'dreem_{subject_id[j]}_{date}']

                    df.replace({'SLEEP-S0':'0','SLEEP-S1':'1','SLEEP-S2':'2','SLEEP-S3':'3','SLEEP-REM':'4','SLEEP-MT':None},inplace=True)
                    df.to_csv(f'{directory}/csv/dreem_{subject_id[j]}_{date}.csv')
                    
                    
data_folder = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/dreem/{subject_id[j]}/'

    
parser=argparse.ArgumentParser()
parser.add_argument('--project_name', type=str)
parser.add_argument('--model', type=str)
args = parser.parse_args(args=['--project_name', 'Dreem', '--model', 'YASA'])

# Check folder
os.chdir(data_folder)

# Set directory path to where the EDF files are located
edf_dir = data_folder + '/edf/'
csv_dir = data_folder + '/txt/csv/'
fft_dir = data_folder + '/fft/'
noise_dir = data_folder + '/noise/'

# # Get list of EDF files in the directory
path_edfs = sorted([file for file in os.listdir(edf_dir)],reverse=True) #if file.startswith('sfsleeproject_test01')]

# path_edf = path_edfs[i]

# # Extract date information from the file name using regular expressions
# file_date = re.search(r'\d{4}-\d{2}-\d{2}', path_edf).group()
# file_date = dt.strptime(file_date, '%Y-%m-%d').date()

# Iterate through each file and check if the target_date is present in the filename
path_edf = None  # Initialize path_edf outside the loop

for pathi in path_edfs:
    file_date = re.search(r'\d{4}-\d{2}-\d{2}', pathi).group()
    print(file_date)
    if file_date:        
        if file_date == target_date:
            path_edf = pathi
            file_date = dt.datetime.strptime(file_date, '%Y-%m-%d').date()
            break

if path_edf:


    # Extract time from the time string
    file_time = re.search(r'\d{2}-\d{2}-\d{2}\[\d{2}-\d{2}\]', path_edf).group()
    file_time = file_time[:8]
    file_time = dt.datetime.strptime(file_time, '%H-%M-%S').time()

    # Check if the time is greater than 19:00:00
    if file_time > dt.datetime.strptime('19:00:00', '%H:%M:%S').time():
        file_date += timedelta(days=1)  # Add one day

    print(f"EDF file {path_edf} was recorded on {file_date.strftime('%Y-%m-%d')}")
    path_edf = 'edf/' + path_edf

    # stage file
    stagefile = f"dreem_{subject_id[j]}_{file_date.strftime('%Y-%m-%d')}.csv"
    if stagefile not in os.listdir(csv_dir):
        print("Stage file not found:", stagefile)
        sys.exit(0)

    path_stages = os.path.join(csv_dir, stagefile)

    eeg_sleep_instance = EEGSleep(project_name=args.project_name)
    acc = eeg_sleep_instance.get_accelerometer(path_edf, preload=True)

    np.save(f'{fft_dir}/df_acc_{subject_id[j]}_{file_date}.npy', acc.get_data())

    epochs, _ = eeg_sleep_instance.preprocess_eeg_data(path_edf, path_stages, preload=True, l_freq=0.75, h_freq=20)


    delta_mark_all, theta_mark_all, alpha_mark_all, beta_mark_all, kurt_mark_all, var_mark_all, mob_mark_all, comp_mark_all, amp_mark_all, diff_mark_all, maxk_amp_all, fftWelch, ch_names = eeg_sleep_instance.compute_noise_matrices(epochs, epochs.info)


    _, noise,mark_all = eeg_sleep_instance.noise_summary(delta_mark_all, theta_mark_all, alpha_mark_all, beta_mark_all, kurt_mark_all, var_mark_all, mob_mark_all, comp_mark_all, amp_mark_all, diff_mark_all, maxk_amp_all, ch_names)

    best_chan, rejected_epochs, final_mark = eeg_sleep_instance.choose_best_electrode(epochs.info, mark_all)

    np.save(f'{fft_dir}/fft_{subject_id[j]}_{file_date}.npy', fftWelch)
    pd.DataFrame(rejected_epochs).to_csv(f'{noise_dir}/noise_{subject_id[j]}_{file_date}.csv')
    np.save(f'{noise_dir}/bestChan_{subject_id[j]}_{file_date}.npy',best_chan)

else:
    print(f'no files from {target_date} for {subject_id[j]}')


