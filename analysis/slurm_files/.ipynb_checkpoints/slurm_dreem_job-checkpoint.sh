#!/bin/bash
#SBATCH --time 1:00:00
#SBATCH --job-name=Sleep_study
#SBATCH --output=ceph/dreem_job%a.out
#SBATCH --array=0-200# Number of tasks/subjects
#SBATCH --mem 400GB

cd /mnt/home/mhacohen/python_files

python dreem_edf2.py $SLURM_ARRAY_TASK_ID