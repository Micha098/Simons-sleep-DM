#!/bin/bash
#SBATCH --time 1:00:00
#SBATCH --job-name=Sleep_study
#SBATCH --output=ceph/dreem_job%a.out
#SBATCH --array=1-10   # Number of tasks/subjects
#SBATCH --mem 20GB

# Load the necessary modules or activate the virtual environment if required
source slurm_files/init_conda.sh

# Change to the directory containing your Python script
cd /mnt/home/mhacohen/python_files

# Call your Python script and pass the subject ID as an argument
python dreem_edf.py $SLURM_ARRAY_TASK_ID $TARGET_DATE

