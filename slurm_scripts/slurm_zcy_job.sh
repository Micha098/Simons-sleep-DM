#!/bin/bash
#SBATCH --time 36:00:00
#SBATCH --job-name=ZCY_disBatch
#SBATCH --output=ceph/zcy_disBatch_job_%j.out
#SBATCH --mem=400GB
#SBATCH --nodes=1
#SBATCH --ntasks=5  # Adjust based on available resources and desired concurrency
#SBATCH --cpus-per-task=1  # Adjust if each task requires multiple CPUs

module load disBatch
TASKS_FILE="/mnt/home/mhacohen/zcy_tasks"

# Run disBatch with the Tasks file
disBatch $TASKS_FILE
