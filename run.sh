#!/bin/bash

module load cuda11/11.0

make test_cpu

srun run.slurm

make time_gpu

srun run.slurm

make test

srun run.slurm

make stream

srun run2.slurm

make memory_info
