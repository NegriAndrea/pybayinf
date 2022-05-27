#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import h5py
from astropy.table import Table
import astropy.units as u
import astropy.cosmology.units as cu

from pybayinf import bayesian_fit
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='Fit a Schechter or a double Schecther'
        ' funcion in galaxy clusters (Negri et al. 2022). Add the flags'
        ' --fit-1Sc and --fit-2Sc to actually perform the fit, if no flag is'
        ' present a check on the data will be performed.')
parser.add_argument('galaxyFile', type=str, help='File containing 2 columns:'
        ' galaxy magnitude (col name mag), cluster ID of the galaxy (col name'
        ' ID)')
parser.add_argument('clusterFile', type=str, help='File containing 2 columns:'
        ' volume of the cluster (col name V), cluster ID (col name ID)')
parser.add_argument('outputFile', type=str,
        help='name of the output file')
parser.add_argument('-m', type=float,
        help='limiting magnitude [default: the faintest galaxy in galaxyFile]')
parser.add_argument('-n', type=int,
        help='numer of threads for computation [%(default)d]', default=4)

parser.add_argument('--fit-1Sch',
        help='perform 1 Schecther fit', action='store_true')
parser.add_argument('--fit-2Sch',
        help='perform 2 Schecther fit', action='store_true')

parser.add_argument('-b', type=int,
        help='burn in points [%(default)d]', default=1000)
parser.add_argument('-l', type=int,
        help='chain lenght [%(default)d]', default=5000)
parser.add_argument('-w', type=int,
        help='number of walkers [%(default)d]', default=30)

parser.add_argument('-o',
        help='overwrite output', action='store_true')

args = parser.parse_args()

gal = Table.read(args.galaxyFile, format='ascii')
cluster = Table.read(args.clusterFile, format='ascii')

data = gal['mag']
dataClusterID = gal['ID']

cID = cluster['ID']
V = cluster['V']

data = data.astype(np.float64)

if args.m is None:
    mag_lim = data.max()
else:
    mag_lim = args.m
    if mag_lim > data.max():
        print('WARNING: the selected magnitude limit is not consistent with'
                ' the data')

if args.o:
    writeMode = 'w'
else:
    writeMode = 'w-'

with h5py.File(args.outputFile+'.h5', writeMode) as ff:
    bayesian_fit(data, dataClusterID, cID, V, mag_lim,
            args.n, args.w, args.l, args.b, ff,
            Sc2=args.fit_2Sch, Sc1=args.fit_1Sch)
