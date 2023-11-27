#!/bin/bash
#SBATCH --time 1:00:00
#SBATCH --job-name=Sleep_study
#SBATCH --output=ceph/dreem_%a.out
#SBATCH --mem 10GB

# Call your Python script and pass the subject ID as an argument
# Loop over subjects
for target_date in {2023-11-17,2023-11-18,2023-11-19,2023-11-20,2023-11-21,2023-11-22,2023-11-23,2023-11-24,2023-11-25,2023-11-26,2023-11-27,2023-11-28}; do
    # Call your Python script and pass the subject ID and date as arguments

    sbatch --export=TARGET_DATE=$target_date slurm_files/slurm_dreem_job.sh
done