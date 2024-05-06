#!/bin/bash
#SBATCH --time 2:00:00
#SBATCH --job-name=YasaDreemScoring
#SBATCH --output=ceph/acc_job_%a.out
#SBATCH --array=2-66 # Number of tasks/subjects
#SBATCH --mem 40GB

# Load the necessary modules or activate the virtual environment if required
source slurm_files/init_conda.sh

# Change to the directory containing your Python script
cd /mnt/home/mhacohen/python_files/

# Iterate over each date for the current task
python Empatica_rawdata_acc.py $SLURM_ARRAY_TASK_ID

