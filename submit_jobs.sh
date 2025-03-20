#!/bin/bash

learning_rates=(0.0001 0.0005 0.001 0.005 0.01 0.05)
weight_decays=(0.0 0.1)

log_file="submitted_jobs.txt"
echo "Submitting jobs at $(date)" > $log_file

for lr in "${learning_rates[@]}"; do
    for wd in "${weight_decays[@]}"; do
        echo "Submitting job with learning_rate=${lr}, weight_decay=${wd}"
        job_id=$(sbatch --parsable slurm/train.sh --learning-rate $lr --weight-decay $wd)
        echo "JobID: ${job_id}, lr=${lr}, wd=${wd}" >> $log_file
    done
done

echo "Submitted $(( ${#learning_rates[@]} * ${#weight_decays[@]} )) jobs"
echo "See $log_file for details"