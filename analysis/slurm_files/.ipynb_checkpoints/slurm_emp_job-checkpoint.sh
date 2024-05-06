#!/bin/bash
#SBATCH --time 1:00:00
#SBATCH --job-name=YasaDreemScoring
#SBATCH --output=ceph/emp_job_%a.out
#SBATCH --array=1-4 # Number of tasks/subjects
#SBATCH --mem 20GB

# Load the necessary modules or activate the virtual environment if required
source slurm_files/init_conda.sh

# Change to the directory containing your Python script
cd /mnt/home/mhacohen/python_files/


# Call your Python script and pass the subject ID as an argument
# enter data in a yyyy-mm-dd format
python Empatica_rawdata.py $SLURM_ARRAY_TASK_ID ''

