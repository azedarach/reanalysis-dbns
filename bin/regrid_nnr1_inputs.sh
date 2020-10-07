#!/bin/bash

# License: MIT

# Regrid NNR1 input fields.

PROJECT_DIR="${HOME}/projects/reanalysis-dbns"
BIN_DIR="${PROJECT_DIR}/bin"
RESULTS_DIR="${PROJECT_DIR}/results/nnr1"
HADISST_RESULTS_DIR="${PROJECT_DIR}/results/hadisst"

REANALYSIS_PERIOD="19480101_20200530"

REGRID_FIELDS="${BIN_DIR}/regrid_cdo.sh"
OVERWRITE="yes"

if test ! -d "$RESULTS_DIR" ; then
  mkdir -p "$RESULTS_DIR"
fi

if test ! -d "$HADISST_RESULTS_DIR" ; then
  mkdir -p "$HADISST_RESULTS_DIR"
fi

slp_datafile="${RESULTS_DIR}/fields/nnr1.slp.${REANALYSIS_PERIOD}.nc"
regridded_slp_datafile=$(echo "$slp_datafile" | sed 's/\.nc/.2.5x2.5.nc/g')

u850_datafile="${RESULTS_DIR}/fields/nnr1.uwnd.850.${REANALYSIS_PERIOD}.nc"
regridded_u850_datafile=$(echo "$u850_datafile" | sed 's/\.nc/.2.5x2.5.nc/g')

u200_datafile="${RESULTS_DIR}/fields/nnr1.uwnd.200.${REANALYSIS_PERIOD}.nc"
regridded_u200_datafile=$(echo "$u200_datafile" | sed 's/\.nc/.2.5x2.5.nc/g')

usfc_datafile="${RESULTS_DIR}/fields/nnr1.uwnd.sig995.${REANALYSIS_PERIOD}.nc"
regridded_usfc_datafile=$(echo "$usfc_datafile" | sed 's/\.nc/.2.5x2.5.nc/g')

vsfc_datafile="${RESULTS_DIR}/fields/nnr1.vwnd.sig995.${REANALYSIS_PERIOD}.nc"
regridded_vsfc_datafile=$(echo "$vsfc_datafile" | sed 's/\.nc/.2.5x2.5.nc/g')

olr_datafile="${RESULTS_DIR}/fields/nnr1.ulwrf.ntat.gauss.${REANALYSIS_PERIOD}.nc"
regridded_olr_datafile=$(echo "$olr_datafile" | sed 's/\.nc/.2.5x2.5.nc/g')

sst_datafile="${HADISST_RESULTS_DIR}/fields/HadISST_sst.nc"
regridded_sst_datafile=$(echo "$sst_datafile" | sed 's/\.nc/.2.5x2.5.nc/g')

if test ! -d "${HADISST_RESULTS_DIR}/fields" ; then
  mkdir -p "${HADISST_RESULTS_DIR}/fields"
fi

files_to_regrid="\
${u850_datafile},${regridded_u850_datafile} \
${u200_datafile},${regridded_u200_datafile} \
${slp_datafile},${regridded_slp_datafile}   \
${usfc_datafile},${regridded_usfc_datafile} \
${vsfc_datafile},${regridded_vsfc_datafile} \
${olr_datafile},${regridded_olr_datafile}   \
${sst_datafile},${regridded_sst_datafile}   \
"

for f in $files_to_regrid ; do

  input_file=$(echo "$f" | cut -d , -f 1)
  output_file=$(echo "$f" | cut -d , -f 2)

  if test -e "$output_file" ; then

    if test "x$OVERWRITE" = "xyes" ; then
      echo "Warning: removing existing output file $output_file"
      rm "$output_file"
    else
      echo "Error: output file $output_file exists"
      exit 1
    fi

  fi

  $REGRID_FIELDS "$input_file" "$output_file"

done
