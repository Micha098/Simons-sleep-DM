#!/bin/bash
#SBATCH --time 1:00:00
#SBATCH --job-name=YasaDreemScoring
#SBATCH --output=ceph/temp_%a.out
#SBATCH --array=1#-67 # Number of tasks/subjects
#SBATCH --mem 40GB

# Load the necessary modules or activate the virtual environment if required

# Change to the directory containing your Python script
cd /mnt/home/mhacohen/python_files


# Call your Python script and pass the subject ID as an argument
python Empatica_rawdata_temp.py $SLURM_ARRAY_TASK_ID
