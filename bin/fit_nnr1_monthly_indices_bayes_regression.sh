#!/bin/bash

# Fit NNR1 monthly indices using linear Gaussian model.

PROJECT_DIR="${HOME}/projects/reanalysis-dbns"
BIN_DIR="${PROJECT_DIR}/bin"
RESULTS_DIR="${PROJECT_DIR}/results"

REANALYSIS="nnr1"
FREQUENCY="monthly"
MODEL="bayes_regression"

OUTPUT_DIR="${RESULTS_DIR}/${REANALYSIS}/models/nc"

PYTHON="python"
FIT_MODEL="${BIN_DIR}/fit_${REANALYSIS}_${FREQUENCY}_indices_${MODEL}.py"

N_CHAINS="4"

$PYTHON "$FIT_MODEL" $@ "$OUTPUT_DIR"
