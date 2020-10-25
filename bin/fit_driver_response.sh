#!/bin/bash

# Fit Bayes regression model to driver-response system.

PROJECT_DIR="${HOME}/projects/reanalysis-dbns"
BIN_DIR="${PROJECT_DIR}/bin"
OUTPUT_DIR="${PROJECT_DIR}/results/driver-response"

if test ! -d "$OUTPUT_DIR" ; then
  mkdir -p "$OUTPUT_DIR"
fi

PYTHON="python"
RUN_FIT="${BIN_DIR}/fit_driver_response_example.py"

$PYTHON "$RUN_FIT" "$@" "$OUTPUT_DIR"
