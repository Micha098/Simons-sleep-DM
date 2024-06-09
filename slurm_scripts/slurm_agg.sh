#!/bin/bash
#SBATCH --time 04:00:00
#SBATCH --job-name=agg_emp_disBatch
#SBATCH --output=ceph/agg_emp_disBatch_job_%j.out
#SBATCH --mem=400GB
#SBATCH --nodes=1
#SBATCH --ntasks=20  # Adjust based on available resources and desired concurrency
#SBATCH --cpus-per-task=1  # Adjust if each task requires multiple CPUs

module load disBatch
TASKS_FILE="/mnt/home/mhacohen/aggEmp_tasks"

# Run disBatch with the Tasks file
disBatch $TASKS_FILE
