#!/bin/bash
#SBATCH --time 04:00:00
#SBATCH --job-name=hypno_disBatch
#SBATCH --output=ceph/hypno_disBatchh_job_%j.out
#SBATCH --mem=400GB
#SBATCH --nodes=1
#SBATCH --ntasks=100  # Adjust based on available resources and desired concurrency
#SBATCH --cpus-per-task=1  # Adjust if each task requires multiple CPUs

module load disBatch
TASKS_FILE="/mnt/home/mhacohen/dreem_hypno_tasks"

# Run disBatch with the Tasks file
disBatch $TASKS_FILE
