# Simons-sleep-DM
Data management scripts for Simons Sleep Project 

1. [Abstract](#abstract)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Citation](#citation)


## Abstract

This repository holds the codebase, scripts and files for the creation and managment of the Simons Sleep Project pilot repository.

In this project we collected data from at-home sleep recordings using an EEG headband (Dreem; Beacon bio-siganls Ltd., a under-the-mattress pressures sensor (Withings ) and a multi-sensor smartwatch (EmbracePlus).

## Requirements
he following environment and libraries are required:

1. Python 3.8 or newer
2. AWS CLI 
3. An operating system with support for Python and the above libraries (tested on Linux Ubuntu 20.04 and macOS Catalina)

Other OS/Python distributions are expected to work.

## Installation
### Prepare new environment Using Conda:

This project provides a sleep.yml file which contains all the necessary Python package dependencies.
> conda env create -f sleep.yml

This command reads the sleep.yml file, sets up the sleep_study environment with all specified packages and their versions.

Activate the environment before running the script with:
> conda activate sleep

## Project Structure

The project operates on a hierarchical pipeline logic, beginning with the synchronization of database and AWS bucket data, followed by preprocessing and data harmonization. The final steps involve processing the data to infer basic sleep and activity measures.

The primary orchestrators for the Dreem and Empatica data are the empatica_sync.py and dreem_sync.py scripts, respectively. These scripts include embedded slurm commands to process user data, iterating over participant dates and performing various data management tasks, such as timezone harmonization, typo correction, and raw data analysis for deriving metrics like activity counts and sleep/wake classifications.

# Empatica Sync Script
The empatica_sync.py script is responsible for pulling data from the AWS cloud for Empatica devices, organizing it according to participant and date, and initiating subsequent processing steps. It uses AWS CLI commands for data synchronization and schedules daily tasks to update and process new data.

Key steps include:

1. Synchronizing Empatica device data from an AWS S3 bucket to a local directory.
2. Running slurm batch jobs to preform preprocessing and hramonization of the the data
3. aggregating the data and generating summarized sleep data reports.

This script ensures that all Empatica data is current and correctly allocated, facilitating the comprehensive analysis of participant sleep patterns.
The General structure of the code works on hirracical logic of pipline of set of actions starting from the syncronization of the data in the databese exising in the cluser with the device data the is on the AWS bucket for each of the different devices. The next step after the syncronization is preprocessing and harmonization of the data and finaly there are proccesing steps to infer some basic sleep and activity measures form the data. 


### Loop over specified dates
for target_date in {2023-11-17,2023-11-18,2023-11-19,2023-11-20,2023-11-21,2023-11-22,2023-11-23,2023-11-24,2023-11-25,2023-11-26,2023-11-27,2023-11-28}; do
    # Call your Python script and pass the subject ID and date as arguments
    sbatch slurm_files/init_conda.sh
    sbatch --export=TARGET_DATE=$target_date slurm_files/slurm_zcy_job.sh
done

## 1.2 Iteration Over Subjects (i.e. slurm_files/slurm_zcy_job.sh)
#!/bin/bash
#SBATCH --time 1:00:00
#SBATCH --job-name=Sleep_study
#SBATCH --output=ceph/dreem_job%a.out
#SBATCH --array=1-10   # Number of tasks/subjects
#SBATCH --mem 20GB

### Load the necessary modules or activate the virtual environment if required
source slurm_files/init_conda.sh

### Change to the directory containing your Python script
cd /mnt/home/mhacohen/python_files

### Call your Python script and pass the subject ID as an argument
python empatica_zcy.py $SLURM_ARRAY_TASK_ID $TARGET_DATE

# 2. Python Files

## i.e. empatica_zcy.py
This Python script processes data for the Dreem sleep study. It reads Empatica accelerometer data, performs preprocessing, and generates activity counts.

from avro.datafile import DataFileReader
from avro.io import DatumReader
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
import json
import os
import sys
import glob 
import pytz
import numpy as np
import pandas as pd
import re
import subprocess
from scipy import signal
import matplotlib.dates as mdates
import seaborn as sns
import datetime
from sklearn import preprocessing



i = int(sys.argv[1]) -1 
date = sys.argv[2]
execute_preprocessing = True

subject_id = pd.read_csv('/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/subjects_ids.csv')['id'].tolist()

#define path, folders, user 
participant_data_path = '/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/aws_data/1/1/participant_data/' # path to the folder that contains folder for each date
output_folder = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/raw_data/{subject_id[i]}/acc/' #output folder

#accelerometer data 
dfs = pd.DataFrame()
dates = sorted(os.listdir(participant_data_path), reverse=True) #all date-folders available 
#dates.remove('.DS_Store')
filename = f'empatica_acc_{subject_id[i]}_'+date+'.csv'

if (filename in sorted(os.listdir(output_folder),reverse=True)):
    
    if (os.path.getsize(output_folder+filename) > 250000000):
        print(f'{filename} already exist')
        execute_preprocessing = False

elif execute_preprocessing:
    print(date)
    folder = os.listdir(participant_data_path+date) # list folders (for each user) within the date-folde
    subfolder = glob.glob(os.path.join(participant_data_path, f'*{date}*/*{subject_id[i]}*//raw_data/v6/')) #path to avro files (within date->within user)    
    if  subfolder != []:
        if os.path.isdir(subfolder[0]):
            files = os.listdir(subfolder[0]) #list of avro files
            files = np.sort(files).tolist() # rearrange files in a chronological manner
            for ff in files: #loop through files to read and store data
                avro_file = subfolder[0]+ff
                reader = DataFileReader(open(avro_file, "rb"), DatumReader())
                schema = json.loads(reader.meta.get('avro.schema').decode('utf-8'))
                data = []
                for datum in reader:
                    data = datum
                reader.close()

                acc = data["rawData"]["accelerometer"] #access specific metric 
                startSeconds = acc["timestampStart"] / 1000000 # convert timestamp to seconds
                timeSeconds = list(range(0,len(acc['x'])))
                if acc["samplingFrequency"] == 0:
                    acc["samplingFrequency"] = 64;
                timeUNIX = [t/acc["samplingFrequency"]+startSeconds for t in timeSeconds]
                delta_physical = acc["imuParams"]["physicalMax"] - acc["imuParams"]["physicalMin"]
                delta_digital = acc["imuParams"]["digitalMax"] - acc["imuParams"]["digitalMin"]
                acc['x'] = [val*delta_physical/delta_digital for val in acc["x"]]
                acc['y'] = [val*delta_physical/delta_digital for val in acc["y"]]
                acc['z'] = [val*delta_physical/delta_digital for val in acc["z"]]

                df_acTot = pd.concat([pd.DataFrame(timeUNIX), pd.DataFrame(acc['x']),pd.DataFrame(acc['y']),pd.DataFrame(acc['z'])],axis = 1)

                if not df_acTot.empty:
                    df_acTot.columns = ['time','x','y','z']
                    dfs = pd.concat([dfs,df_acTot])
            dfs=dfs.reset_index(drop = True)
            dfs.to_csv(output_folder+f'empatica_acc_{subject_id[i]}_'+date+'.csv')
            dfs = pd.DataFrame()
            print('finished preprocessing '+date)
    else:
        print(f'subfolfer {date} empty')


input_folder = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/raw_data/{subject_id[i]}/acc/' #output folder
output_folder =  f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/empatica/measure_data/{subject_id[i]}/zcy/' 


files = sorted(os.listdir(input_folder), reverse= True)
#folders.remove('.DS_Store')
#load accelerometery datas

target_date = sys.argv[2]

path_acc = None  # Initialize path_edf outside the loop

for pathi in files:
    try:
        file_date = re.search(r'\d{4}-\d{2}-\d{2}', pathi).group()
        print(file_date)
    except:
        continue
    if file_date:        
        if file_date == target_date:
            path_acc = pathi
            file_date = dt.datetime.strptime(file_date, '%Y-%m-%d').date()
            break

if path_acc:

    FileName = f'{input_folder}{path_acc}'
    New_fileName = f'zcy{path_acc[12:]}'
    
    if (New_fileName in sorted(os.listdir(output_folder),reverse=True)):
        print(f'{New_fileName} already exist')

        if (os.path.getsize(output_folder+New_fileName) > 50000):
            print(f'{New_fileName} a accaptable size file exist')
            sys.exit()

    else:

        if (f'{subject_id[i]}' in path_acc) & (FileName.endswith('.csv')):

            data = pd.read_csv(FileName,index_col=0)
            #try:
            data['date'] = data['time'].apply(lambda x: dt.datetime.utcfromtimestamp(x))
            #except KeyError:
            #    try:
            #        data['date'] = data['timestamp'].apply(lambda x: datetime.datetime.utcfromtimestamp(x))
            #    except KeyError:
            #        pass

            data = data.set_index('date', drop = True)

            # slice data to begin with nearest hour
            startTime = data.index[0]
            #startTimeNew = startTime.replace(microsecond=540000, second=0, minute=0, hour=startTime.hour+1)
            if startTime.minute< 15:
                startTimeNew = startTime.replace(microsecond=0, second=0, minute=15, hour=startTime.hour)
            elif (startTime.minute>= 15 & startTime.minute < 30):
                startTimeNew = startTime.replace(microsecond=0, second=0, minute=30, hour=startTime.hour)
            elif (startTime.minute>= 30 & startTime.minute < 45):
                startTimeNew = startTime.replace(microsecond=0, second=0, minute=45, hour=startTime.hour)
            else:
                startTimeNew = startTime.replace(microsecond=0, second=0, minute=45, hour=startTime.hour+1)

            data2 = data.loc[data.index>=startTimeNew,]

            # set parameters (like GGIR)
            hb = 3
            lb = 0.25
            n = 2
            sf = 64

            Wc = np.zeros(2)
            Wc[0] = lb/(sf/2) 
            Wc[1] = hb/(sf/2)
            Wc

            b,a = signal.butter(n, Wc, 'bandpass')
            # Calibrate the data by subtracting the mean
            data2 -= data2.mean()
            data2['y'] = signal.lfilter(b, a, data2['y'])

            timeVec = data2.resample('5S', convention = 'start').mean().drop(['time','x','y','z'],axis=1)
            for i in timeVec.index:
                mask = (data2.index<=i) & (data2.index > i-datetime.timedelta(seconds=5))
                d = data2.loc[mask]

                if d.shape[0] == 320:
                    y = d['y'].values
                    #y=signal.lfilter(b,a,d['y'])
                    Ndat = len(y)
                    #change the values of y < 0.01 to 0
                    y[np.abs(y)<0.01]=0


                    # Create the vector of 1 and -1
                    Vec = np.ones_like(y)
                    Vec[y<0]=-1

                    tmp = abs(np.sign(Vec[1:Ndat])-np.sign(Vec[0:Ndat-1]))*0.5
                    tmp = np.append(tmp[0],tmp)
                    cs = np.cumsum(np.append(0,tmp))
                    slct = np.arange(0, len(cs), 5*sf)
                    x3 = np.diff(cs[np.round(slct)])
                    timeVec.loc[i,'ZCY']=x3[0]
                else:
                    timeVec.loc[i,'ZCY']=np.nan

            timeVec = timeVec.resample('30S', convention = 'start').sum()

            #timeVec = timeVec.resample('60S', convention = 'start').sum()
            timeVec.to_csv(f'{output_folder}/{New_fileName}')

            print(f'finished processing {New_fileName}')
# 3. Jupyter Notebook

## 3.1 Daily_test_script.ipynb
This Jupyter notebook generates visualizations from the output of the Python scripts.

for user in subject_ids:
        # try:
            directory_fft = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/dreem/{user}/fft'
            directory_noise = f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/dreem/{user}/noise'
            directory_stages= f'/mnt/home/mhacohen/ceph/Sleep_study/SubjectsData/dreem/{user}/txt/csv'


            for fft in sorted(os.listdir(directory_fft)):
                    if fft.startswith("fft"):


                        date = re.search(r"\d{4}-\d{2}-\d{2}", fft).group()
                        noisy_epochs_file = f'noise_{user}_{date}.csv'
                        stages_file = f'dreem_{user}_{date}.csv'
                        electrode_file = f'bestChan_{user}_{date}.npy'
                        acc_file = f'df_acc_{user}_{date}.npy'
                        
                        stages = pd.read_csv(directory_stages+f'/{stages_file}')
                        noisy_epochs = pd.read_csv(directory_noise + f'/{noisy_epochs_file}', index_col='Unnamed: 0')

                        # Load FFT file
                        best_elec = np.load(directory_noise+f'/{electrode_file}')
                        data = (np.load(directory_fft + f'/{fft}'))
                        best_elec_indices = best_elec[:, np.newaxis]  # Add a new axis to match the array shape

                        # Use the indices to select the desired channel for each epoch
                        selected_channels = data[np.arange(len(best_elec)), best_elec_indices.flatten(), :]
                        fft_data = pd.DataFrame(selected_channels)
                        # finish epoching accelerometer data
                        acc = np.load(directory_fft + f'/{acc_file}')
                        epoch_duration = 30  # seconds
                        samples_per_epoch = int(sfreq * epoch_duration)

                        # Calculate the total number of epochs
                        total_epochs = acc.shape[1] // samples_per_epoch
                        acc = acc[:, :total_epochs * samples_per_epoch]
                        # Reshape the data into epochs
                        acc_epoched = acc.reshape((total_epochs, samples_per_epoch))
                        percentiles = np.percentile(acc_epoched, [5, 95], axis=1)


                        power_all = fft_data.iloc[:,allfreqidx[0]:allfreqidx[1]]
                        # take out noisy epochs
                        power_all_1 = power_all.copy()
                        channel1 = pd.DataFrame(data[:,0, :]).iloc[:,allfreqidx[0]:allfreqidx[1]]
                        channel2 = pd.DataFrame(data[:,1, :]).iloc[:,allfreqidx[0]:allfreqidx[1]]
                        channel3 = pd.DataFrame(data[:,2, :]).iloc[:,allfreqidx[0]:allfreqidx[1]]
                        channel4 = pd.DataFrame(data[:,3, :]).iloc[:,allfreqidx[0]:allfreqidx[1]]

                        power_all.loc[noisy_epochs.values] = np.nan


                        # std_threshold = 3
                        # std = np.std(power_all, axis=0)
                        # power_all[np.abs(power_all - np.mean(power_all)) > std_threshold * std] = np.nan
                        # power_all[np.abs(power_all) > 600 ] = np.nan

                        power_filt = power_all.copy()
                        power_nan = np.isnan(power_filt)

                        #intepolate missing values across columns (frequencies)
                        for i in power_filt.columns:
                            nan = power_nan.loc[:, i]
                            if nan.all():  # Check if the whole column is NaN
                                power_filt.loc[:, i] = 0    

                            else:
                                power_filt.loc[nan, i] = np.interp(np.flatnonzero(nan), np.flatnonzero(~nan), power_filt.loc[~nan, i])


                        # Compute the absolute power by approximating the area under the curve
                        delta_power = pd.DataFrame(simps(power_filt.iloc[:,deltaidx[0]:deltaidx[1]], dx=freq_res))
                        delta_power_1 = simps(power_all_1.iloc[:,deltaidx[0]:deltaidx[1]], dx=freq_res)
                        delta_smoothed = gaussian_filter(delta_power, sigma=10, mode='wrap')
                        
                        stages['time'] = pd.to_datetime(stages['Time [hh:mm:ss]']) # Convert time to datetime
                        # resamples epochs to round 30 seconds
                        stages = stages.set_index(stages['time'])  # Set the time as the index
                        stages = stages.resample('30S').first() # Change '30T' to your desired time interval
                        stages.time = pd.to_datetime(stages.index).time
                        stages.reset_index(inplace=True,drop=True)

                        fig, ax = plt.subplots(figsize=(10, 5))

                        ax.plot(delta_power, label='after interpolation', alpha=0.8)
                        ax.plot(delta_power_1, label='before interpolation', alpha=0.4)
                        ax.set_title(f'Delta power: {date} {user}', fontsize=15)
                        ax.set_xlabel('Time', fontsize=15)
                        ax.set_ylabel('Average Delta Power', fontsize=15)
                        ax.set_xlim(-10,len(delta_power))
                        ax.set_ylim(-30,130)
                        ax.set_yticks(ax.get_yticks()[2:-3])

                        ax.set_xticklabels(stages.time.iloc[ax.get_xticks()[:-1]])

                        # Plot Dreem Sleep staging on the first y-axis
                        ax2 = ax.twinx()

                        # Create a second y-axis
                        ax2.tick_params(axis='y', pad=5)
                        ax2.plot(-stages['Sleep Stage'].replace({1:2,2:3,3:4,4:1}), label='Dreem Sleep staging', alpha=0.8, lw=2, color='black')
                        ax2.scatter(stages[np.isnan(stages['Sleep Stage'])].index,stages[np.isnan(stages['Sleep Stage'])]['Sleep Stage'].replace({np.nan:1}), label='Noise',marker = 's', color='red')

                        # Set the y-axis label for the left side
                        ax2.set_ylabel('Sleep Stage', fontsize=15, loc= 'top')
                        ax2.set_yticks([0, -1, -2, -3, -4])  # Set y-ticks to match the stages
                        ax2.set_yticklabels(['Awake', 'Rem ','N1', 'N2', 'Deep' ])

                        lines, labels = ax.get_legend_handles_labels()
                        lines2, labels2 = ax2.get_legend_handles_labels()
                        ax2.legend(lines + lines2, labels + labels2, loc='lower left')
                        # Use 'viridis' colormap with reduced saturation
                        ax2.set_ylim(-25,2)
                        ax2.tick_params(axis='y', pad=20)  # Increase the spacing between tick labels and axis
                        
                        # Plot accelerometer data
                        
                        percentiles = np.percentile(acc_epoched*1e7, [5, 95], axis=1)
                        ax4 = ax.twinx()

                        ax4.fill_between(range(len(percentiles[0])), percentiles[0], percentiles[1])
                        ax4.set_ylim(0,14)
                        ax4.set_yticks([])
                        ax4.set_ylabel('Accelerometer', fontsize=12, color = 'C0', rotation=0)
                        ax4.yaxis.set_label_coords(1,0.7)

                        fig, axs = plt.subplots(4, 1, figsize=(12.8, 8))
                        fig.subplots_adjust(hspace=0.8)

                        ## Plot the spectrograms for each channel
                        for axi, channel, title in zip(axs, [channel1, channel2, channel3, channel4], ['EEG F7-O1', 'EEG F8-O2', 'EEG F8-O1', 'EEG F7-O2']):
                            grouped_power = channel.groupby(np.arange(len(channel.columns)) // 4, axis=1).sum()
                            im = axi.imshow(grouped_power.T, aspect='auto', cmap='inferno_r', origin='lower', extent=(0, len(grouped_power), 0, 30), vmin=0, vmax=30)
                            axi.set_title(title)
                            axi.set_ylabel('Frequency (Hz)')

                            plt.colorbar(im,label='Power')
                            axi.set_ylim(0,30)
                            #axi.set_xticks([])

                            axi.set_xticklabels(ax.get_xticklabels()[1:])
                        plt.show()
                        plt.close()


# Usage

Run the Slurm scripts to iterate over dates and subjects.
The Python scripts process the data and generate output files.
Use the Jupyter notebook to visualize the results.

