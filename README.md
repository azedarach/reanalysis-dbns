Climate DBNs from reanalyses
============================
[![DOI](https://zenodo.org/badge/301906792.svg)](https://zenodo.org/badge/latestdoi/301906792)

This repository contains the code used
for a study of dynamic Bayesian networks
applied to reanalysis data.

To install from source, run:

    python setup.py install

It is recommended that the package be installed into a custom
environment. For example, to install into a custom conda
environment, first create the environment via

    conda create -n reanalysis-dbns-env python=3.8
    conda activate reanalysis-dbns-env

The package may then be installed using

    cd /path/to/package/directory
    python setup.py install

Optionally, a set of unit tests may be run by executing

    python setup.py test
