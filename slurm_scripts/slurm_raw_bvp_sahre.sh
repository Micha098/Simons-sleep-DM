#!/bin/bash
#SBATCH --time 36:00:00
#SBATCH --job-name=bvp_raw_job__disBatch
#SBATCH --output=ceph/out/bvp_raw_disBatch_job_%j.out
#SBATCH --mem=400GB
#SBATCH --nodes=1
#SBATCH --ntasks=10  # Adjust based on available resources and desired concurrency
#SBATCH --cpus-per-task=1  # Adjust if each task requires multiple CPUs

module load disBatch
TASKS_FILE="/mnt/home/mhacohen/bvp_share_tasks"

# Run disBatch with the Tasks file
disBatch $TASKS_FILE
