#!/bin/bash
#SBATCH --time 1:00:00
#SBATCH --job-name=Sleep_study
#SBATCH --output=ceph/emp_hr_%a.out
#SBATCH --mem 20GB

# Call your Python script and pass the subject ID as an argument
# Loop over subjects
for target_date in {2023-12-06,2023-12-07,2023-12-08,2023-12-09,2023-12-10,2023-12-11,2023-12-12,2023-12-13,2023-12-14,2023-12-15,12-16,12-17}; do
    # Call your Python script and pass the subject ID and date as arguments
    sbatch slurm_files/init_conda.sh

    sbatch --export=TARGET_DATE=$target_date slurm_files/slurm_hr_job.sh
done