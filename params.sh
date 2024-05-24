#!/bin/bash
export JOBS="4" # 2D multipliers

export UW_NAME="Dev_23_02_2024" # uw version name

export SCALING_TYPE=1  # 1=weak, 2=strong

export UW_MODEL="timed_model_SolC"  #  "SolDB3d" for dim3, though penalty method probably needed for q1.

export PICKLENAME=None      #"SolDB3d_Gadi_1e-11.pickle" #"conv_test_results_high_res_tighter_take2.pickle"  # set to "None" to disable conv testing
export UW_ENABLE_IO="0"     # Jan 2024 - not used
###

export WALLTIME="00:10:00"
export ACCOUNT="el06"
export QUEUE="normal"       # normal or express

export UW_ORDER=1           # Jan 2024 - not used

export UW_DIM=2             # 2 or 3
export SCALING_BASE=32      # use 32 for for 2D

# Test style - UW_MAX_ITS (+ve, recommended >100): Fixed work, (-ve): Accuracy (UW_SOL_TERANCE is used)
export UW_MAX_ITS=50 # set to negaive for accuracy test, positive for fixed iterative work irrepective of result fidelity
export UW_SOL_TOLERANCE=1e-5

# unused 09/01/2024
export UW_PENALTY=-1.   # set to negative value to disable penalty
###
