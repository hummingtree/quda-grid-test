#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
export QUDA_RESOURCE_PATH="./cache"

mpirun -n 2 ./summit.x --mpi 1.1.1.2 --grid 12.12.12.16 --comms-overlap

