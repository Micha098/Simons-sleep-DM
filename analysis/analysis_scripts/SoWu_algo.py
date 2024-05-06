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
from datetime import datetime as dt
from datetime import timedelta
from datetime import time
from utilities import project_data_dir
from eeg_sleep import EEGSleep
import argparse

i = int(sys.argv[1])

# for user_dir, user_name in zip(['Micha','Ilan','Sachin'],['mh','id','sc']):
user_dir = sys.argv[2]
user_name = sys.argv[3]

data_folder = f'/mnt/ceph/users/mhacohen/data/Dreem/algo/{user_dir}/'
output_folder = f'/mnt/ceph/users/mhacohen/data/Dreem/algo/{user_dir}/'


parser=argparse.ArgumentParser()
parser.add_argument('--project_name', type=str)
parser.add_argument('--model', type=str)
args = parser.parse_args(args=['--project_name', 'Dreem', '--model', 'YASA'])

# Set directory path to where the EDF files are located
edf_dir = f'/mnt/ceph/users/mhacohen/data/Dreem/{user_dir}/edf/'
csv_dir = f'/mnt/ceph/users/mhacohen/data/Dreem/{user_dir}/csv/'
fft_dir = data_folder
noise_dir = data_folder

# Check folder
os.chdir(edf_dir)

# Get list of EDF files in the directory
path_edfs = sorted([file for file in os.listdir(edf_dir) if file.endswith('.edf')],reverse=True)

path_edf = path_edfs[i]
print(path_edf)
# Extract date information from the file name using regular expressions
file_date = re.search(r'\d{4}-\d{2}-\d{2}', path_edf).group()
file_date = dt.strptime(file_date, '%Y-%m-%d').date()

if (user_name == 'id') & (file_date == dt(2023, 5, 16).date()):
    print("Skipping file:", path_edf)
    sys.exit(0)

# Extract time from the time string
file_time = re.search(r'\d{2}-\d{2}-\d{2}\[\d{2}-\d{2}\]', path_edf).group()
file_time = file_time[:8]
file_time = dt.strptime(file_time, '%H-%M-%S').time()

# Check if the time is greater than 19:00:00
if file_time > dt.strptime('19:00:00', '%H:%M:%S').time():
    file_date += timedelta(days=1)  # Add one day

if i == 1:
    algo_stats = pd.DataFrame()
    algo_stats.to_csv(f'{output_folder}/algo_stats_{user_name}.csv')

print(f"EDF file {path_edf} was recorded on {file_date.strftime('%Y-%m-%d')}")
# skip the file if is in output directory
#if f'fft_{user_id}_{file_date}.csv' in os.listdir(output_folder):

#    print('File already exists:', path_edf)

# stage file
stagefile = f"dreem_{user_name}_{file_date.strftime('%Y_%m_%d')}.csv"
if stagefile not in os.listdir(csv_dir):
    print("Stage file not found:", stagefile)
    sys.exit(0)

path_stages = os.path.join(csv_dir, stagefile)
stages = pd.read_csv(path_stages)
stages.rename(columns={'Time [hh:mm:ss]':'time'},inplace =True)
stages.drop(['Unnamed: 0'], axis =1,inplace =True)
stages.to_csv(f'{path_stages}')


eeg_sleep_instance = EEGSleep(project_name=args.project_name)


epochs, croppedData = eeg_sleep_instance.preprocess_eeg_data(path_edf, path_stages, preload=True, l_freq=0.75, h_freq=20)

print(epochs)
token = None
df_pred = eeg_sleep_instance.compute_Alg_stages(croppedData, epochs.info,token, path_edf)
df_pred['Dreem'] = stages['Sleep Stage']

df_pred.replace({'W':0,'R':4,'N1':1,'N2':2,'N3':3}, inplace=True)

df_pred['max_conf'] = df_pred.iloc[:, 1::2].idxmax(axis=1)

# Extract the corresponding channel values based on the 'max_conf' column
df_pred['yasa'] = df_pred.apply(lambda row: row[row['max_conf'].replace('_conf', '')], axis=1)

# Identify the index where the first 10 consecutive epochs are greater than 0
so_yasa = df_pred.index[(df_pred['yasa'] > 0).rolling(window=10, min_periods=1).sum() == 10].min()
so_dreem = df_pred.index[(df_pred['Dreem'] > 0).rolling(window=10, min_periods=1).sum() == 10].min()

wu_yasa = df_pred.index[(df_pred['yasa'] > 0).rolling(window=10, min_periods=1).sum() == 10].max()
wu_dreem = df_pred.index[(df_pred['Dreem'] > 0).rolling(window=10, min_periods=1).sum() == 10].max()

delta_mark_all, theta_mark_all, alpha_mark_all, beta_mark_all, kurt_mark_all, var_mark_all, mob_mark_all, comp_mark_all, amp_mark_all, diff_mark_all,maxk_amp_all, fftWelch, ch_names = eeg_sleep_instance.compute_noise_matrices(epochs, epochs.info,sd_crt = 2)


_, noise,mark_all = eeg_sleep_instance.noise_summary(delta_mark_all, theta_mark_all, alpha_mark_all, beta_mark_all, kurt_mark_all, var_mark_all, mob_mark_all, comp_mark_all, amp_mark_all, diff_mark_all,maxk_amp_all, ch_names)


best_chan, rejected_epochs, final_mark = eeg_sleep_instance.choose_best_electrode(epochs.info, mark_all)

np.save(f'{output_folder}/mark_all_{user_name}_{file_date}.npy', mark_all)

np.save(f'{output_folder}/fft_{user_name}_{file_date}.npy', fftWelch)
pd.DataFrame(rejected_epochs).to_csv(f'{noise_dir}/noise_{user_name}_{file_date}.csv')

rejected_epochs = pd.read_csv(os.path.join(data_folder,f'noise_{user_name}_{file_date}.csv'))['0']

np.save(f'{output_folder}/bestChan_{user_name}_{file_date}.npy',best_chan)

fftClean = np.load(data_folder + f'/fft_{user_name}_{file_date}.npy')

best_elec = np.load(data_folder + f'/bestChan_{user_name}_{file_date}.npy')

fftClean = pd.DataFrame(fftClean[np.arange(len(best_elec)), best_elec,:])

so_algo, wu_algo, delta_smoothed, _, _, _, _, peaks, so_flag,  wu_flag, ratioAT_smoothed = eeg_sleep_instance.SoWu_algo_check_conditions(epochs.info, fftClean, rejected_epochs, nan_flag = 1, prom_flag = 0)

algo_stats = pd.DataFrame([[file_date, so_algo, wu_algo,peaks, so_flag, wu_flag,so_yasa,wu_yasa,so_dreem,wu_dreem]], columns=['Date','so_algo','wu_algo','peaks','so_flag', 'fa_flag','so_yasa','wu_yasa','so_dreem','wu_dreem'])

df = pd.read_csv(f'{output_folder}/algo_stats_{user_name}.csv')

df = pd.concat([df, algo_stats], ignore_index=True)
df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

print(algo_stats)

df.to_csv(f'{output_folder}/algo_stats_{user_name}.csv')
