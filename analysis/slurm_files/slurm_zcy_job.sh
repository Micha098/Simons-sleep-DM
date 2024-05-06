#!/bin/bash
#SBATCH --time 4:00:00
#SBATCH --job-name=ZCY
#SBATCH --output=ceph/zcy_job_%a.out
#SBATCH --array=0-87 # Number of tasks/subjects
#SBATCH --mem 400GB



# Change to the directory containing your Python script
cd /mnt/home/mhacohen/python_files/

    # Call your Python script and pass the subject ID and date as arguments
python empatica_zcy.py $SLURM_ARRAY_TASK_ID

