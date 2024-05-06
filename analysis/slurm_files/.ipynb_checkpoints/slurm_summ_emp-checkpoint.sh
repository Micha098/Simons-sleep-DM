#!/bin/bash
#SBATCH --time 1:00:00
#SBATCH --job-name=aggregated_data
#SBATCH --output=ceph/summ_nights%a.out
#SBATCH --array=0-200# Number of tasks/subjects
#SBATCH --mem 400GB

# Load the necessary modules or activate the virtual environment if required
source slurm_files/init_conda.sh


# Change to the directory containing your Python script
cd /mnt/home/mhacohen/python_files/

python summ_emp.py $SLURM_ARRAY_TASK_ID
