#!/bin/bash
#SBATCH --time 10:00:00
#SBATCH --job-name=withings_disBatch
#SBATCH --output=ceph/summ_nights_withings%j.out
#SBATCH --mem=400GB
#SBATCH --nodes=1
#SBATCH --ntasks=30  # Adjust based on available resources and desired concurrency
#SBATCH --cpus-per-task=1  # Adjust if each task requires multiple CPUs

module load disBatch
TASKS_FILE="/mnt/home/mhacohen/withings_tasks"

# Run disBatch with the Tasks file
disBatch $TASKS_FILE
