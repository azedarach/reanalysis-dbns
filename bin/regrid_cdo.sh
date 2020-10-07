#!/bin/bash

# Regrid input file using CDO

CDO="cdo"
REGRID_METHOD="-remapbil"
TARGET_GRID="r144x72"

if test $# -lt 2 ; then
  echo "Error: too few arguments given"
  exit 1
fi

input_file="$1"
output_file="$2"

$CDO ${REGRID_METHOD},${TARGET_GRID} "$input_file" "$output_file"
