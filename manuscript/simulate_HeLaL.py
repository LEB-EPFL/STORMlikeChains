"""Simulate HeLa L telomeres and compute simulated Rg distributions.

This script will simulate HeLa L telomeres and generate radius of
gyration distributions for comparison to the experimental STORM data.

"""

__author__ = 'Kyle M. Douglass'
__email__  = 'kyle.douglass@epfl.ch'

# Ensure the script can access PolymerPy during development.
# Add PolymerPy path to PYTHONPATH during installation.
import sys
if len(sys.argv) > 1:
    if sys.argv[1] == '-d':
        sys.path.append('/home/douglass/src/PolymerPy/')

# The example begins here.
from PolymerPy import PolymerPy

from numpy import ones, append, arange, concatenate, meshgrid, loadtxt
from numpy.random import random

import time

# Define grid of parameter values to simulate
C1, LP1 = meshgrid(  arange(10, 60, 5),         arange(10, 105, 5))
C2, LP2 = meshgrid(  arange(30, 65, 5),         arange(105, 205, 5))
C3, LP3 = meshgrid(arange(60, 100, 10),         arange(10, 220, 20))
C4, LP4 = meshgrid(        array([20]),         arange(110, 210, 20))
C5, LP5 = meshgrid(arange(65, 105, 10), arange(25, 205, 5))
C6, LP6 = meshgrid(arange(60, 100, 10), array([25, 35, 40, 45, 55, 60, 65, 75, 80, 85, 95, 100]))
C7, LP7 = meshgrid(arange(70, 100, 10), array([105, 115, 120, 125, 135, 140, 145, 155, 160, 165, 175, 180, 185, 195, 200, 205]))

C = concatenate((C1.flatten(),
		 C2.flatten(),
		 C3.flatten(),
		 C4.flatten(),
		 C5.flatten(),
		 C6.flatten(),
		 C7.flatten()))
LP = concatenate((LP1.flatten(),
		  LP2.flatten(),
		  LP3.flatten(),
		  LP4.flatten(),
		  LP5.flatten(),
		  LP6.flatten(),
		  LP7.flatten()))

"""Setup the input parameters for the simulation.

numPaths      : int
    Number of paths (i.e. chains) to simulate for each pair of values
    for the linear density and persistence length.
pathLength    : array of float (units of base pairs)
    The number of base pairs in each chain.
linDensity    : array of float (units of base pairs per nanometer)
    The values for the linear packing ratio of base pairs to simulate.
persisLength  : array of float (units of nanometers)
    The persistence length values to simulate.
segConvFactor : float (units of segments per nanometer)
    The number of chain segments per nanometer; used to convert units
    of length from nanometers to chain segments.
nameDB        : string
    The name of the database where the simulation data will be saved.
locPrecision  : float (units of nanometers)
    The localization precision for simulating STORM datasets.
fullSpecParam : bool
    Are the linear densities and persistence lengths defined as linear
    arrays from which a meshgrid will be created? Or, are these
    parameters specified as two linear arrays where the full list of
    simulated values are defined by corresponding pairs of values in
    each array?

    This will usually be set to True, so all one must specify is every
    pair of values that one wants to simulate.
chainSubsamples: int
    How many segments to retain when subsampling the chain. This will
    typically be set to the average number of localizations per
    cluster. Default is -1, which means to keep all the segments.

"""

# Open the data for HeLa S genomic lengths.
with open('HeLaLGenomicLength.txt', 'r') as genomicLengthsFile:
    genomicLengths = loadtxt(genomicLengthsFile)

numPaths     = 100000
basePairDist = genomicLengths[0:numPaths] * 1000 # Convert from kb to bp

simArgs = {'numPaths'        : numPaths,
           'pathLength'      : basePairDist,
           'linDensity'      : C,
           'persisLength'    : LP,
           'segConvFactor'   : 2.5,
           'nameDB'          : 'HeLaL_Simulated_Rg',
           'locPrecision'    : 15, # nm
           'fullSpecParam'   : True,
           'chainSubsamples' : 200}

tic = time.time()

# Unpack the argument dictionary and call the collector.
myCollector = PolymerPy.WLCCollector(**simArgs)
toc = time.time()

print('Total simulation time: %f' % (toc - tic))
