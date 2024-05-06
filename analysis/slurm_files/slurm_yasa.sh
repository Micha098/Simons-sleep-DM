#!/bin/bash
#SBATCH --time 1:00:00
#SBATCH --job-name=YasaDreemScoring
#SBATCH --output=ceph/eeg_%a.out
#SBATCH --array=1-50# Number of tasks/subjects
#SBATCH --mem 10GB

# Load the necessary modules or activate the virtual environment if required
 
# Change to the directory containing your Python script
cd /mnt/home/mhacohen/python_files

# Call your Python script and pass the subject ID as an argument
python Yasa_Dreem.py $SLURM_ARRAY_TASK_ID 'Ilan' 'id'