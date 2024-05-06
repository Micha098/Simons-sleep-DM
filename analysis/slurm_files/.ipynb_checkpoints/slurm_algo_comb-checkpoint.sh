#!/bin/bash
#SBATCH --time 2:00:00
#SBATCH --job-name=YasaDreemScoring
#SBATCH --output=ceph/eeg_%a.out
#SBATCH --array=1-2   # Number of tasks/subjects
#SBATCH --mem 40GB

# Load the necessary modules or activate the virtual environment if required
 
# Change to the directory containing your Python script
cd /mnt/home/mhacohen

# Call your Python script and pass the subject ID as an argument
python YasaUdreem_comb.py $SLURM_ARRAY_TASK_ID 'Micha' 'mh'