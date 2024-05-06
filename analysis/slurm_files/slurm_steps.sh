#!/bin/bash
#SBATCH --time 1:00:00
#SBATCH --job-name=YasaDreemScoring
#SBATCH --output=ceph/steps_%a.out
#SBATCH --array=1-140   # Number of tasks/subjects
#SBATCH --mem 30GB

# Load the necessary modules or activate the virtual environment if required
# Change to the directory containing your Python script
cd /mnt/home/mhacohen

# Call your Python script and pass the subject ID as an argument
python steps_ggir.py $SLURM_ARRAY_TASK_ID 'ilan' '04'

