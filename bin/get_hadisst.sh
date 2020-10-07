#!/bin/sh

# Fetch HadISST data.

WGET="wget"
EXTRACT="gunzip"

HADOBS_URL="https://www.metoffice.gov.uk/hadobs"
DATASET_NAME="hadisst"

DATAFILES="HadISST_sst.nc.gz HadISST_ice.nc.gz"

#_______________________________________________
timestamp() {
    date -u "+%Y-%m-%dT%H:%M:%s%z"
}

#_______________________________________________
create_datafile_url() {
    _url="${HADOBS_URL}/${DATASET_NAME}/data/$1"
}

#_______________________________________________
fetch_datafile() {
    $WGET -O "$2" $1
}

if test $# -lt 1 ; then
  echo "Error: too few arguments"
  echo "Usage: ./$(basename $0) DATA_DIR"
  exit 1
fi

data_dir="$1"

if test ! -d "$data_dir" ; then
    mkdir "$data_dir"
fi

for f in $DATAFILES ; do
    dest="${data_dir}/$f"
    timestamp_file="${data_dir}/$f.timestamp"
    create_datafile_url $f
    src_url="$_url"
    fetch_datafile "$src_url" "$dest"
    status="$?"
    if test "x$status" != "x0" ; then
	     msg="Error: failed to fetch datafile (exit code $status)"
	     echo "$msg"
	     exit "$status"
    fi
    timestamp > "$timestamp_file"

    $EXTRACT "$dest"
done
