#!/bin/bash
#SBATCH --time 1:00:00
#SBATCH --job-name=bvp_raw
#SBATCH --output=ceph/bvp_%a.out
#SBATCH --array=65-85  # Number of tasks/subjects
#SBATCH --mem 500GB

# Load the necessary modules or activate the virtual environment if required
source slurm_files/init_conda.sh

# Change to the directory containing your Python script
cd /mnt/home/mhacohen/python_files/


# Call your Python script and pass the subject ID as an argument
python Empatica_rawdata_bvp.py $SLURM_ARRAY_TASK_ID
