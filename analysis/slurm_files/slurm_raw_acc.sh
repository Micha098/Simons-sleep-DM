#!/bin/bash
#SBATCH --time 12:00:00
#SBATCH --job-name=raw_empatica_data
#SBATCH --output=ceph/raw_acc_%a.out
#SBATCH --array=100-200 # Number of tasks/subjects
#SBATCH --mem 500GB

# Load the necessary modules or activate the virtual environment if required

# Change to the directory containing your Python script
cd /mnt/home/mhacohen/python_files


# Call your Python script and pass the subject ID as an argument
python Empatica_rawdata_acc.py $SLURM_ARRAY_TASK_ID
