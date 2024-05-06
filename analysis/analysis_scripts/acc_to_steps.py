import pandas as pd
import os
import re
import sys
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import datetime
from sklearn import preprocessing
import numpy as np
from scipy.signal import find_peaks
import math
import warnings

warnings.filterwarnings("ignore")

i = int(sys.argv[1])
user_name = sys.argv[2]

steps_df = pd.DataFrame(columns=['time', 'steps'])

# Load accelerometer data from CSV
input_folder = f'/mnt/home/mhacohen/ceph/data/Empatica/raw_data/acc/{user_name}/' 
output_folder =  f'/mnt/home/mhacohen/ceph/data/Empatica/steps/{user_name}/' 

files = os.listdir(input_folder)
#folders.remove('.DS_Store')
#load accelerometery data

FileName = f'{input_folder}{files[i]}'
New_fileName = f'steps{files[i][12:]}'

if (New_fileName not in sorted(os.listdir(output_folder),reverse=True)) & (FileName.endswith('.csv')):
    if (user_id in files[i]) & (FileName.endswith('.csv')):

    data = pd.read_csv(FileName)

    data['date'] = data['time'].apply(lambda x: datetime.datetime.utcfromtimestamp(x))
    data = data.set_index('date', drop = True)

    # Extract data for each axis
    xdata = data['x']
    ydata = data['y']
    zdata = data['z']

    # Combine the data from the three axes into a single magnitude scalar value
    accel_mag = np.sqrt(xdata**2 + ydata**2 + zdata**2)

    mean_accel = np.mean(accel_mag)
    std_accel = np.std(accel_mag)

    # Set the threshold as a multiple of the standard deviation
    threshold_multiplier = 4  # Adjust this value as needed
    threshold = mean_accel + threshold_multiplier * std_accel

    # Detect peaks in the combined acceleration data
    #peaks, _ = find_peaks(accel_mag, height=threshold)

    # Calculate the number of steps in 60-second epochs
    epoch_duration = pd.Timedelta(seconds=60)
    start_time = data.index.min()
    end_time = data.index.max()
    current_time = start_time

    while current_time <= end_time:
        # Select the data for the current epoch
        epoch_df = data[(data.index >= current_time) & (data.index < current_time + epoch_duration)]

        # Combine the data from the three axes into a single magnitude scalar value
        accel_mag = np.sqrt(epoch_df['x']**2 + epoch_df['y']**2 + epoch_df['z']**2)

        # Set the threshold as a multiple of the standard deviation
        threshold_multiplier = 4  # Adjust this value as needed
        threshold = accel_mag.mean() + threshold_multiplier * accel_mag.std()

        # Detect peaks in the combined acceleration data using the dynamic threshold
        peaks, _ = find_peaks(accel_mag, height=threshold)

        # Count the number of steps
        num_steps = len(peaks)

        # Print or save the result for the current epoch
        steps_df = pd.concat([steps_df, pd.DataFrame({'time': [current_time], 'steps': [num_steps]})], ignore_index=True)

        # Move to the next epoch
        current_time += epoch_duration

    # Export the results DataFrame to a CSV file
    steps_df.to_csv(f'{output_folder}{New_fileName}',index=False)
