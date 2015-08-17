"""Run the simulation for the wild type Hela L experiment.
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

from numpy import array, ones, append, arange, concatenate, meshgrid, loadtxt
from numpy.random import random

import time

# Define two separate square grids of parameter pair values.
C1, LP1 = meshgrid(  arange(10, 60, 5),         arange(10, 105, 5))
C2, LP2 = meshgrid(  arange(30, 65, 5),         arange(105, 205, 5))
C3, LP3 = meshgrid(arange(60, 100, 10),         arange(10, 220, 20))
C4, LP4 = meshgrid(        array([20]),         arange(110, 210, 20))

C = concatenate((C1.flatten(), C2.flatten(),
                 C3.flatten(), C4.flatten()))
LP = concatenate((LP1.flatten(), LP2.flatten(),
                  LP3.flatten(), LP4.flatten()))

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
    pair of values that one wants to simulate. This allows for non-rectangular
    arrays of parameters.

"""
# Load the genomic length distribution from the file
with open('HeLaLGenomicLength.txt', 'r') as genomicLengthsFile:
    genomicLengths = loadtxt(genomicLengthsFile)
    
numPaths     = 100000
basePairDist = genomicLengths[0:numPaths] * 1000 # Convert from kb to bp

simArgs = {'numPaths'      : numPaths,
           'pathLength'    : basePairDist,
           'linDensity'    : C,
           'persisLength'  : LP,
           'segConvFactor' : 2.5,
           'nameDB'        : 'simData_HelaL_WT_' + PolymerPy.dateStr,
           'locPrecision'  : 15,
           'fullSpecParam' : True}

tic = time.time()

# Unpack the argument dictionary and call the collector.
myCollector = PolymerPy.WLCCollector(**simArgs)
toc = time.time()

print('Total simulation time: %f' % (toc - tic))
