#!/bin/bash
# This script sets up the conda environment and required environment variables.

# Activate the 'tdlu' conda environment
source /fs/nexus-scratch/tuxunlu/miniconda3/bin/activate pointcept

# Set environment variables
export PYTHONPATH="/fs/nexus-scratch/tuxunlu/git/event-based-deraining:$PYTHONPATH"
module load gcc/14.2.0
module load cuda/12.1.1