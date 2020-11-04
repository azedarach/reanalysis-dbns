#!/bin/bash

# Fit JRA-55 monthly indices using linear Gaussian model.

PROJECT_DIR="${HOME}/projects/reanalysis-dbns"
BIN_DIR="${PROJECT_DIR}/bin"
RESULTS_DIR="${PROJECT_DIR}/results"

REANALYSIS="jra55"
FREQUENCY="monthly"
MODEL="bayes_regression"

OUTPUT_DIR="${RESULTS_DIR}/${REANALYSIS}/models/nc"

PYTHON="python"
FIT_MODEL="${BIN_DIR}/fit_${REANALYSIS}_${FREQUENCY}_indices_${MODEL}.py"

$PYTHON "$FIT_MODEL" $@ "$OUTPUT_DIR"
