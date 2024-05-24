#!/bin/bash

# the follow load the full software stack and running environment on gadi
source /home/157/jg0883/install-scripts/gadi_install.sh
env
#cat timed_model_${UW_MODEL}.py
cat ${UW_MODEL}.py

echo ""
echo "---------- Running Job ----------"
echo ""

export TIME_LAUNCH_MPI=`date +%s%N | cut -b1-13`
#mpiexec -n ${NTASKS} bash -c "TIME_LAUNCH_PYTHON=\`date +%s%N | cut -b1-13\` python3 timed_model_3D.py"
#mpiexec -n ${NTASKS} bash -c "TIME_LAUNCH_PYTHON=\`date +%s%N | cut -b1-13\` python3 timed_model_2D.py"

#mpiexec -n ${NTASKS} bash -c "TIME_LAUNCH_PYTHON=\`date +%s%N | cut -b1-13\` python3 timed_model_${UW_MODEL}.py -log_view :${UW_MODEL}_SCALING_TYPE_${SCALING_TYPE}_NPROCS_${NTASKS}_${UW_DIM}D.txt:ascii_flamegraph"

mpiexec -n ${NTASKS} -x LD_PRELOAD=libmpi.so bash -c "TIME_LAUNCH_PYTHON=\`date +%s%N | cut -b1-13\` python3 ${UW_MODEL}.py -log_view :${UW_MODEL}_SCALING_TYPE_${SCALING_TYPE}_NPROCS_${NTASKS}_${UW_DIM}D.txt"

#mpiexec -n ${NTASKS} bash -c "TIME_LAUNCH_PYTHON=\`date +%s%N | cut -b1-13\` python3 ${UW_MODEL}.py -log_view :${UW_MODEL}_SCALING_TYPE_${SCALING_TYPE}_NPROCS_${NTASKS}_${UW_DIM}D.xml:ascii_xml"

# create flame graph
#mpiexec -n ${NTASKS} bash -c "TIME_LAUNCH_PYTHON=\`date +%s%N | cut -b1-13\` python3 ${UW_MODEL}.py -log_view :${UW_MODEL}_SCALING_TYPE_${SCALING_TYPE}_NPROCS_${NTASKS}_${UW_DIM}D.txt:ascii_flamegraph"
