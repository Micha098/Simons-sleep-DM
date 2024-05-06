#!/bin/bash
#SBATCH --time 1:00:00
#SBATCH --job-name=Sleep_study
#SBATCH --output=ceph/dreem_%a.out
#SBATCH --mem 20GB

# Call your Python script and pass the subject ID as an argument
# Loop over subjects
for target_date in {2023-11-06,2023-11-07,2023-11-08,2023-11-09,2023-11-10}; do
    # Call your Python script and pass the subject ID and date as arguments

    sbatch --export=TARGET_DATE=$target_date slurm_files/slurm_dreem_job.sh
done