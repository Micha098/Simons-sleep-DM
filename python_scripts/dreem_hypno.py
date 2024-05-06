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
from dateutil import parser as dt_parser  # Aliased to avoid conflicts
import pytz
from utilities import project_data_dir
from eeg_sleep import EEGSleep
import argparse
import mne
import shutil

def get_dreem_hypno(subject_id, directory):
    
    df_dreem = pd.DataFrame()

    for filename in sorted(os.listdir(directory)):
        try:
            if filename.startswith(f"dreem_{subject_id}") and (filename.endswith(f".csv")):

                try:
                    date = re.search(r"\d{4}-\d{2}-\d{2}", filename).group()

                except:

                    date = re.search(r"\d{4}_\d{2}_\d{2}", filename).group().replace('_','-')

                df = pd.read_csv(f'{directory}/{filename}')


                if 'time' in df.columns:

                    df.rename(columns={'time':'timestamp'},inplace=True)

                elif 'Time [hh:mm:ss]' in df.columns:

                    df.rename(columns={'Time [hh:mm:ss]':'timestamp'},inplace=True)

                df.timestamp= pd.to_datetime(df.timestamp,format ="%H:%M:%S")

                date_temp = dt.datetime.strptime(date, '%Y-%m-%d')
                end_date = date_temp.strftime("%Y-%m-%d")


                # add correct date to the Time column
                if (df.timestamp[0].strftime("%H:%M:%S") <= "23:59:59" and df.timestamp[0].strftime("%H:%M:%S") >= "10:00:00"):
                    start_date = (date_temp - timedelta(days=1)).strftime("%Y-%m-%d")

                else: start_date = end_date

                dates= [start_date,end_date]

                start_time = df.timestamp[0]
                end_time = df.timestamp.iloc[-1]

                start_date = date_temp.strftime("%Y-%m-%d")



                if (df.timestamp[0] > pd.to_datetime("2023-03-12"+' '+"02:00:00")) and (df.timestamp[0] < pd.to_datetime("2023-11-05"+' '+"02:00:00")):

                    df.timestamp = pd.date_range(start=f'{dates[0]} {start_time}-04:00', end=f'{dates[1]} {end_time}-04:00', freq='30S')
                else:
                    df.timestamp = pd.date_range(start=f'{dates[0]} {start_time}-05:00', end=f'{dates[1]} {end_time}-05:00', freq='30S')

                df.timestamp = pd.to_datetime(df.timestamp).dt.tz_localize(None)
                # make timestamp an index
                df.set_index(df['timestamp'],inplace=True, drop=True)
                df.sort_index(inplace=True)
                df.drop_duplicates(inplace=True)

                df.drop(columns={'timestamp'}, inplace=True)
                #df.drop(columns={'Event','Duration[s]'},inplace=True)
                
                df['Sleep Stage'] = df['Sleep Stage'].astype(float)
                df['rec_time'] = len(df['Sleep Stage'])*0.5/60
                df['rec_quality'] = len(df['Sleep Stage'].dropna())/len(df['Sleep Stage'])
                so = df['Sleep Stage'].rolling(window=20).apply(lambda x: (x > 0).all()).idxmax()
                fa =  df['Sleep Stage'].iloc[::-1].rolling(window=20).apply(lambda x: (x > 0).all()).idxmax()

                if df['rec_time'] > 10:
                    df['rec_time'] = 10
                    fa =  df['Sleep Stage'].iloc[::10*120].rolling(window=20).apply(lambda x: (x > 0).all()).idxmax()

                    
                # Adding Ilan's criterion for FA
                fa_criteria = (df['Sleep Stage'].iloc[::-1] == 0).rolling(window=180).sum() > 80
                if fa_criteria.any():  
                    fa = fa_criteria.idxmax()
                    
                df['so'] = so
                df['fa'] = fa
                
                df['tst'] = (df.loc[so:fa, 'Sleep Stage'] != 0).sum() * 0.5  # Assuming each row is 30 seconds
                df['waso'] = (df.loc[so:fa, 'Sleep Stage'] == 0).sum() * 0.5  # Assuming each row is 30 seconds
                if df.tst[0] < 180 :
                     df[['waso','tst']] = None

                df_dreem = pd.concat([df,df_dreem]).sort_values(by='timestamp')
        
        except Exception as e:
            print(f"Error processing data for subject_id {subject_id}: {e}")

    if not isinstance(df_dreem.index, pd.DatetimeIndex):
        df_dreem.index = pd.to_datetime(df_dreem.index)

    df_dreem.index = df_dreem.index.round('30s')

    # If you want to set the rounded timestamp as the new index
    #df_dreem.set_index('timestamp', inplace=True)
    locals()[f'df_dreem_{subject_id}'] = df_dreem
    locals()[f'df_dreem_{subject_id}']['subject'] = subject_id

    return df_dreem

j = int(sys.argv[1])

subject_id = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['id'].tolist()
subject_tz = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['tz'].tolist()
tzs_str = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['tz_str'].tolist()

# Convert Dreem Hypmnogram to desired format 

directory = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/{subject_id[j]}/hypno/'
output_folder2 = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/data_share/{subject_id[j]}/dreem/hypno/'
output_folder =f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/{subject_id[j]}/hypno/'

if not os.path.isdir(directory):
        os.makedirs(directory)

if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

if not os.path.isdir(output_folder2):
        os.makedirs(output_folder2)

#check folder
os.chdir(directory)

# remove old processed files 
for file in directory:
    if file.startswith('dreem_'):
        so.remove(os.path.join(directory,file))

date_list=[]
date = None
if len (os.listdir(directory)) > 0:
    for filename in os.listdir(directory):
        if 'hypnogram' in filename:
            
            try:
            
                datetime_str = filename.split('@')[1].split('.')[1]
                datetime_str= datetime_str.split('_')[1]


                # Parse the datetime string into a datetime object, including the timezone
                if '[05-00]' in datetime_str:
                    print(filename)
                    datetime_str.replace("[05-00]", "-05:00")

                datetime_obj = dt_parser.parse(datetime_str)
                # Define the target timezone
                target_tz = pytz.timezone(tzs_str[j])
                utc_fix = False

                if datetime_obj.tzinfo != target_tz:
                    utc_fix = True
                    datetime_obj = datetime_obj.astimezone(target_tz)
                    # print(f"Time adjusted to: {target_tz}")

                file_time = datetime_obj.time()
                file_tz = datetime_obj.tzinfo


                 # Check if the time adjustment crosses over to the next day
                if file_time > dt.datetime.strptime('19:00:00', '%H:%M:%S').time():
                    datetime_obj += timedelta(days=1)  # Add one day

                file_date = datetime_obj.date()

                if filename.endswith("hypnogram.txt"):

                    with open(f"{directory}/{filename}") as file:
                        lines = file.readlines()
                        # Find the line with "idorer Time" and extract the date
                        for i, line in enumerate(lines):
                            if "Scorer Time" in line:

                                date = file_date
                                time_dt = file_time

                                date_list.append(date)

                                new_filename = f"dreem_{subject_id[j]}_{date}.txt"
                                # Delete all lines before the table
                                lines = lines[(i+2):]
                                break
                        # Write the remaining lines to a new file with the new name
                        if date:
                            with open(f"{directory}/{new_filename}", "w") as new_file:
                                new_file.writelines(lines)

                            locals()[f'dreem_{subject_id[j]}_{date}'] = pd.read_csv(f"{directory}/{new_filename}",sep= "\t")
                            df = locals()[f'dreem_{subject_id[j]}_{date}']

                            df.replace({'SLEEP-S0':'0','SLEEP-S1':'1','SLEEP-S2':'2','SLEEP-S3':'3','SLEEP-REM':'4','SLEEP-MT':None},inplace=True)
                            # df.rename(columns = {'Time [hh:mm:ss]':'time'},inplace=True)

                            df['time'] = pd.to_datetime(date.strftime('%Y-%m-%d') + ' ' + df['Time [hh:mm:ss]'], format='%Y-%m-%d %H:%M:%S')
                            df['time'] = df['time'].dt.tz_localize(file_tz.zone)

                            df.drop('Time [hh:mm:ss]', axis=1, inplace=True)
                            
                            reorder = ['Sleep Stage','time','Event','Duration[s]']

                            df = df[reorder]

                            os.remove(directory+filename) # delete the old foramt files
                            
                            new_filename =  f"dreem_{subject_id[j]}_{date}.csv"
                            #df.to_csv(f'{output_folder}/dreem_{subject_id[j]}_{date}.txt', sep='\t')
                            if new_filename in os.listdir(output_folder):
                                    print('oh no, duplicated file!')
                            df.to_csv(so.path.join(output_folder,new_filename), index=False)
                            df.to_csv(so.path.join(output_folder2,new_filename), index=False)
            
            except Exception as e:
                print(f'{subject_id[j]} {e}')
                continue
            
## code for hypnogram foramting

# subject_id = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['id'].tolist()
# subject_tz = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['tz'].tolist()
# tzs_str = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['tz_str'].tolist()

# for j in range(len(subject_id)):
#     # Convert Dreem Hypmnogram to desired format 

#     directory = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/{subject_id[j]}/hypno/'
#     output_folder2 = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/dreem/{subject_id[j]}/txt/csv/'
#     output_folder =f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/raw_data/harmonized_data/{subject_id[j]}/hypno/csv/'

#     if not os.path.isdir(directory):
#             os.makedirs(directory)

#     if not os.path.isdir(output_folder):
#             os.makedirs(output_folder)

#     if not os.path.isdir(output_folder2):
#             os.makedirs(output_folder2)

#     #check folder
#     os.chdir(directory)


#     date_list=[]
#     date = None
#     if len (os.listdir(directory)) > 0:

#         for filename in os.listdir(directory):
#             if 'hypnogram' in filename:
#                 try:
#                     datetime_str = filename.split('@')[1].split('.')[1]
#                     datetime_str= datetime_str.split('_')[1]


#                     # Parse the datetime string into a datetime object, including the timezone
#                     if '[05-00]' in datetime_str:
#                         print(filename)
#                         datetime_str.replace("[05-00]", "-05:00")

#                     datetime_obj = dt_parser.parse(datetime_str)
#                     # Define the target timezone
#                     target_tz = pytz.timezone(tzs_str[j])
#                     utc_fix = False

#                     if datetime_obj.tzinfo != target_tz:
#                         utc_fix = True
#                         datetime_obj = datetime_obj.astimezone(target_tz)
#                         # print(f"Time adjusted to: {target_tz}")

#                     file_time = datetime_obj.time()
#                     file_tz = datetime_obj.tzinfo


#                      # Check if the time adjustment crosses over to the next day
#                     if file_time > dt.datetime.strptime('19:00:00', '%H:%M:%S').time():
#                         datetime_obj += timedelta(days=1)  # Add one day

#                     file_date = datetime_obj.date()

#                     if filename.endswith("hypnogram.txt"):

#                         with open(f"{directory}/{filename}") as file:
#                             lines = file.readlines()
#                             # Find the line with "idorer Time" and extract the date
#                             for i, line in enumerate(lines):
#                                 if "Scorer Time" in line:

#                                     date = file_date
#                                     time_dt = file_time

#                                     date_list.append(date)

#                                     new_filename = f"dreem_{subject_id[j]}_{date}.txt"
#                                     # Delete all lines before the table
#                                     lines = lines[(i+2):]
#                                     break
#                             # Write the remaining lines to a new file with the new name
#                             if date:
#                                 with open(f"{directory}/{new_filename}", "w") as new_file:
#                                     new_file.writelines(lines)

#                                 locals()[f'dreem_{subject_id[j]}_{date}'] = pd.read_csv(f"{directory}/{new_filename}",sep= "\t")
#                                 df = locals()[f'dreem_{subject_id[j]}_{date}']

#                                 df.replace({'SLEEP-S0':'0','SLEEP-S1':'1','SLEEP-S2':'2','SLEEP-S3':'3','SLEEP-REM':'4','SLEEP-MT':None},inplace=True)
#                                 # df.rename(columns = {'Time [hh:mm:ss]':'time'},inplace=True)

#                                 df['time'] = pd.to_datetime(date.strftime('%Y-%m-%d') + ' ' + df['Time [hh:mm:ss]'], format='%Y-%m-%d %H:%M:%S')
#                                 df['time'] = df['time'].dt.tz_localize(file_tz.zone)



#                                 os.remove(directory+filename) # delete the old foramt files

#                                 df.to_csv(f'{output_folder}/dreem_{subject_id[j]}_{date}.txt', sep='\t')
#                                 df.to_csv(f'{output_folder}/dreem_{subject_id[j]}_{date}.csv', index=False)
#                                 df.to_csv(f'{output_folder2}/dreem_{subject_id[j]}_{date}.csv', index=False)

#                 except Exception as e:
#                     print(f'{subject_id[j]} {e}')
#                     continue

