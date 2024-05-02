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
import cProfile
import time as timer



command = [
    'sbatch', '/mnt/home/mhacohen/slurm_files/slurm_raw_acc.sh',
]
subprocess.run(command)

timer.sleep(120 * 60)  # 60 minutes * 60 seconds

command = [
    'sbatch', '/mnt/home/mhacohen/slurm_files/slurm_raw_bvp.sh',
]
subprocess.run(command)

timer.sleep(120 * 60)  # 60 minutes * 60 seconds


command = [
    'sbatch', '/mnt/home/mhacohen/slurm_files/slurm_raw_eda.sh',
]
subprocess.run(command)
timer.sleep(60 * 60)  # 60 minutes * 60 seconds


command = [
    'sbatch', '/mnt/home/mhacohen/slurm_files/slurm_raw_gyro.sh',
]
subprocess.run(command)
timer.sleep(60 * 60)  # 60 minutes * 60 seconds


command = [
    'sbatch', '/mnt/home/mhacohen/slurm_files/slurm_raw_temp.sh',
]
subprocess.run(command)