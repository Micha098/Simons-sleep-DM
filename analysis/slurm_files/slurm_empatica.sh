#!/bin/bash
#SBATCH --time 1:00:00
#SBATCH --job-name=Sleep_study
#SBATCH --output=ceph/empatica_%a.out
#SBATCH --mem 10GB

# Change to the directory containing your Python script

# Call your Python script and pass the subject ID as an argument
# Loop over subjects
# Loop over dates (adjust the start and end dates accordingly)
for target_date in {2023-11-10,2023-11-11,2023-11-12,2023-11-13,2023-11-14,2023-11-15,2023-11-16,2023-11-17,2023-11-18}; do
    # Call your Python script and pass the subject ID and date as arguments

    sbatch --export=TARGET_DATE=$target_date slurm_files/slurm_emp_job.sh
done
