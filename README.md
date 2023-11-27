# Simons-sleep-DM
Data management scripts for Simons Sleep Project 
# Data Management for Sleep Study

## Overview
This repository contains scripts and files for managing data related to a sleep study. The data processing involves Slurm scripts for iteration over dates and subjects, Python scripts for data processing, and a Jupyter notebook for generating visualizations.

# 1. Slurm Scripts

## 1.1 Iteration Over Dates (i.e. `slurm_files/slurm_zcy_job.sh`)

#!/bin/bash
#SBATCH --time 1:00:00
#SBATCH --job-name=Sleep_study
#SBATCH --output=ceph/emp_zcy_%a.out
#SBATCH --mem 20GB

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


# 3. Jupyter Notebook

## 3.1 Daily_test_script.ipynb
This Jupyter notebook generates visualizations from the output of the Python scripts.

# Usage

Run the Slurm scripts to iterate over dates and subjects.
The Python scripts process the data and generate output files.
Use the Jupyter notebook to visualize the results.

