#!/bin/bash
#PBS -N gpu_a3
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -e errorfile.txt
#PBS -o logfile.txt
#PBS -q gpuq

# Change to the directory where the job was submitted
tempdir=$HOME/scratch/job
mkdir -p $tempdir
cd $tempdir
cp -R $PBS_O_WORKDIR/* .

# Load necessary modules (adjust as needed for your system)
module load cuda11.4
nvcc main.cu -o main

# Run the test script
# bash run.sh
./run_testcases.sh

# Copy generated files back to the current directory
cp -r * $PBS_O_WORKDIR/
rm -r $tempdir
#mv * $PBS_O_WORKDIR/
#rmdir $tempdir

# Print job details
echo "Job completed at: $(date)"
echo "Executed on node: $(hostname)"