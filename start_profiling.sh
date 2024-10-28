#!/bin/bash

PROJECTDIR="/home/ggeorge/jinr/dayabay-model"
VENVDIR=".venv"

DAGFLOWDIR="/home/ggeorge/jinr/dag-flow"

# update profiling with current code changes (instead of git)
# [!] This is a temprorary solution before profiling branch is merged. I am going to delete this file soon.
cp ${DAGFLOWDIR}/dagflow/tools/profiling/*.py ${PROJECTDIR}/dagflow/tools/profiling/.

# run profiling
export PYTHONPATH=$PROJECTDIR
source "${PROJECTDIR}/${VENVDIR}/bin/activate"
python3 "${PROJECTDIR}/scripts/run_profiling.py"


# TODO: remove this file
