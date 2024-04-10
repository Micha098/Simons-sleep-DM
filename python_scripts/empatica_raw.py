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
from Empatica_rawdata_acc import acc_raw_data
from Empatica_rawdata_bvp import bvp_raw_data
from Empatica_rawdata_temp import temp_raw_data
from Empatica_rawdata_eda import eda_raw_data


i = int(sys.argv[1]) 

acc_raw_data(i)

print(i)

acc_bvp_data(i)

print(i)

acc_eda_data(i)

print(i)

acc_temp_data(i)

print(i)
