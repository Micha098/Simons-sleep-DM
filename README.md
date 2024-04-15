# Simons-sleep-DM
Data management scripts for the Simons Sleep Project 

1. [Abstract](#abstract)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Citation](#citation)


## Abstract

This repository holds the codebase, scripts and files for the creation and managment of the Simons Sleep Project data repository.

In this project we collected at-home sleep recordings using an EEG headband (Dreem; Beacon bio-siganls), an under-the-mattress sensor (Withings), and a multi-sensor smartwatch (EmbracePlus).

## Requirements
The following environment and libraries are required to run the code:

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

## Initial Project Data Structure - Downloaded from device servers

The data recorded by each of the three devices exists on the device company server and is organized by username (assigned to each participant by the research assisstant when onboarding). Our code begins with download/synchronization of data from each of the three AWS buckets containing this data (one per device) into a folder named "SubjectData", which contains the same data structure as provided by the device company through their AWS buckets, with a sub-folder per device:
![raw_data](https://github.com/Micha098/Simons-sleep-DM/assets/107123518/f5e32b5b-4adb-49d3-b4e1-d3144f1d0464)

## Target Project Data Structure for data sharing

The data in "SubjectData" is processed and harmonized across devices to create a new data structure that enables data sharing. This data is organized in a folder named "data_share" that contains a sub folder per participant, named according to their internal user id (SPARK RIDs), with each participant folder containing a sub folder per device with their harmonized data. Harmonized data contains data timestamps that have been adjusted to reflect the timezone the participant was in when recording, and data files are split into 24 hour periods per date. Bellow is a tree digram that demonstrates the orgnization of the data at the end of the data processing and harmonization:
![data_share](https://github.com/Micha098/Simons-sleep-DM/assets/107123518/ce2a49b8-7102-48ce-badb-22c47a539847)

The primary code files for synching, restructuring, and processing the Dreem, Withings, and Empatica data are the dreem_sync.py, withings_synch.py, and empatica_sync.py scripts, respectively. These scripts include embedded slurm commands to download and process user data, iterating over recording dates and performing various data management tasks, such as mapping device usernames to subject ids, adjusting data timestamps to the participant's timezone, and correction of data acquisition mistakes (e.g., mistakes in the allocation of device usernames). Below is a detailed explanation of each code file per device.

# Empatica Sync Script
The empatica_sync.py script is responsible for pulling data from the AWS cloud for Empatica devices, organizing it according to participant and date, and initiating subsequent processing steps. It uses AWS CLI commands for data synchronization and schedules daily tasks to update and process new data.

This script ensures that all Empatica data is current and correctly allocated, facilitating the comprehensive analysis of participant sleep patterns.
The General structure of the code works on hirracical logic of pipline of set of actions starting from the syncronization of the data in the databese exising in the cluser with the device data the is on the AWS bucket for each of the different devices. The next step after the syncronization is preprocessing and harmonization of time zone of the data.

Key steps include:

1. Synchronizing Empatica device data from an AWS S3 bucket to a local directory.
2. Running slurm batch jobs to preform preprocessing and hramonization of the the data


- **AWS S3 Data Sync**: Sets environment variables for AWS S3 access and synchronizes data from an S3 bucket to a local directory and Uses the AWS CLI command to sync data from the specified S3 bucket to a local path.
  
```
%sync_command = f"aws s3 sync {os.environ['ACCESS_URL']} {os.environ['LOCAL_PATH']} --region us-east-1"
subprocess.run(sync_command, shell=True)
subprocess.run(f"{sync_command} > output.txt", shell=True)
```

### Data Preparation
- **Subject Data Retrieval**: This step extracts subject IDs and their respective time zones from a CSV file at a specified path, ensuring accurate data handling for each subject.



### Aggregated Data Processing
- **Slurm Job Submission**: The aggregated data script submits a Slurm job to process summary measures from the Empatica directory. It combines these measures into a single table per day in the format `empatica_measures_{subject_id}_{date}.csv`.
- **Measurements Processed**: The measures included in this process are wear detection, sleep detection, activity count, step count, heart rate, and respiratory rate. Importantly, since not all summary data were released by Empatica at the same time (i.e. "respiratory rate"), some participents have only some of the measures meantioned above.
- 
- **Time Zone Handling**: Since data on the Empatica server is uploaded in UTC (00:00), the data for each participant is concatenated into one long file per subject. It is then adjusted to match each participant's local time zone before being split again into individual daily files for each subject.

### Raw Data Processing
- **Slurm Job Submission**: A second Slurm job processes raw data from the Empatica directory, producing individual daily files adjusted to each subject's time zone.

- **Raw Data Processed**: This process handles raw data including accelerometer, gyroscope, blood volume pulse (BVP), electrodermal activity (EDA), and temperature. Similar to summary data, the availability of certain types of raw data (e.g., gyroscope) may vary among participants. Data sampling frequencies range from 1 to 64 Hz.
- **Preprocessing**: Raw data are stored in Avro files segmented into 15-minute intervals, formatted as `1-1-{sub_id}_{UNIX timestamp}.avro`, where the timestamp marks the start of the 15-minute period. The script extracts and concatenates these data into one array per day, filling any gaps with NaN values.
- **Time Zone Handling**: As with summary data, raw data uploaded in UTC at 00:00 are concatenated into two-day segments. These are then adjusted to match each participant's local time zone and subsequently split into individual daily files.


A more detailed explantion regarding Emaptica measures, and some preproccing code examples could be found in the Ematica documenation:

/https://manuals.empatica.com/ehmp/careportal/data_access/v2.4e/en.pdf

# Dreem Sync Script

This script handles the synchronization of sleep data from Dreem devices, processes the data, and prepares summary reports. Below are the key functionalities implemented in the script:

#### AWS Data Pull
- **AWS S3 Data Sync**: Sets environment variables for AWS S3 access and synchronizes data from an S3 bucket to a local directory and Uses the AWS CLI command to sync data from the specified S3 bucket to a local path.
  
```
%sync_command = f"aws s3 sync {os.environ['ACCESS_URL']} {os.environ['LOCAL_PATH']} --region us-east-1"
subprocess.run(sync_command, shell=True)
subprocess.run(f"{sync_command} > output.txt", shell=True)
```
#### Data Allocation
- **Unique Case Allocation**: Calls the `dreem_allocation()` function to prepare a list of cases of "unique cases" for data saved incorrectly under the wrong Dreem user id or worng device.
- **subject ids mapping and Allocation**: Since every Dreem_id matches few different participent ids at different dates {that, is different subjects use the same device and id at diiferent dates} The script calls a dictionary table that translates betweeen Dreem-id and dates to the subject ids. the code then allcoates the appropiate dreem files to the folders of the correct subjects.

#### Data Allocation

- **Unique Case Allocation**: Executes the dreem_allocation() function, which identifies and corrects data that may have been saved under incorrect Dreem user IDs or devices. 
- **subject ids mapping and Allocation**: Since every Dreem_id is associated with few different participent ids at different dates  The script allocates the appropriate Dreem files to the folders of the correct subjects, ensuring that each set of data is associated with the right participant for accurate analysis.
   The script calls a dictionary table that translates betweeen Dreem-id and dates to the subject ids. the code then allcoates the appropiate dreem files to the folders of the correct subjects.
Subject IDs Mapping and Allocation: Manages the complexities of device sharing among participants by using a dictionary table that maps Dreem IDs and dates to corresponding subject IDs. The script allocates the appropriate Dreem files to the folders of the correct subjects, ensuring that each set of data is associated with the right participant for accurate analysis.

#### Slurm Job Management
- **Initial Slurm Job Submission**: Submits a job to process Dreem hypnogram data (`slurm_dreem_hypno.sh`), followed by a 10-minute wait.
- **Further Data Processing**: Submits an additional Slurm job (`slurm_dreem_job.sh`) for further data processing, followed by a 60-minute wait to ensure completion.

#### Reporting and Error Handling
- **Error Handling**: Includes robust error handling to catch and log issues during data processing, ensuring transparency in case of failures.
- **Report Generation**: Aggregates data from multiple sources (Withings, Empatica, Dreem) for each subject. Summarizes the number of files processed and valid nights/days recorded for each device type and participant.

#### Final Output
- **Merged Reports**: Combines individual reports into comprehensive tables that list nights and valid measurements per subject. These are further classified by subject type (`ASD` or `NASD`) and saved in a structured format for easy access and analysis.
- **Nightly and Daily Reports**: Generates final reports detailing valid measurement days and nights for each participant, facilitating further analysis on sleep patterns and device efficiency.

