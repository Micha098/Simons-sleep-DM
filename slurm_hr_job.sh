#!/bin/bash
#SBATCH --time 2:00:00
#SBATCH --job-name=YasaDreemScoring
#SBATCH --output=ceph/hr_%a.out
#SBATCH --array=1-6   # Number of tasks/subjects
#SBATCH --mem 40GB

# Load the necessary modules or activate the virtual environment if required
source slurm_files/init_conda.sh

# Change to the directory containing your Python script
cd /mnt/home/mhacohen/python_files/

# Call your Python script and pass the subject ID as an argument
python empatica_hr.py $SLURM_ARRAY_TASK_ID $TARGET_DATE
