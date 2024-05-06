#!/bin/bash
#SBATCH --time 2:00:00
#SBATCH --job-name=YasaDreemScoring
#SBATCH --output=ceph/hr_%a.out
#SBATCH --array=0-87 # Number of tasks/subjects
#SBATCH --mem 400GB
source slurm_files/init_conda.sh

# Change to the directory containing your Python script
cd /mnt/home/mhacohen/python_files/

# Iterate over each date for the current task
python empatica_hr.py $SLURM_ARRAY_TASK_ID
