"""PolymerPy Example: Simulate a number of wormlike chains.

This example script shows how to setup the the wormlike chain
collector and run a simulation. The collector object takes a range of
input parameters and generates wormlike chain ensembles for pairs of
values for the packing ratio and persistence length. The radius of
gyration of each chain conformation in the ensemble is saved in a
NumPy database (NumPyDB). This is simply a class that wraps around
Python's CPickle utility.

Additionally, a single molecule localization experiment is simulated
by specifying a non-zero localization precision. This takes each
monomer in a chain conformation and displaces it randomly in each
direction according to a Gaussian distribution whose standard
deviation is equivalent to the localization precision.

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

from numpy import ones, append, arange, concatenate, meshgrid
from numpy.random import random

import time

# Define two separate square grids of parameter pair values.
C1, LP1 = meshgrid(arange(10, 60, 5), arange(10, 105, 5))
C2, LP2 = meshgrid(arange(30, 65, 5), arange(105, 205, 5))

C  = concatenate((C1.flatten(),  C2.flatten()))
LP = concatenate((LP1.flatten(), LP2.flatten()))

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
# Create a random numbers for the number of base pairs in each chain.
numPaths = 10
basePairDist = 24000 * (random(numPaths) - 0.5) + 27000

simArgs = {'numPaths'        : numPaths,
           'pathLength'      : basePairDist,
           'linDensity'      : C,
           'persisLength'    : LP,
           'segConvFactor'   : 2.5,
           'nameDB'          : 'example_WLC_DB',
           'locPrecision'    : 10,
           'fullSpecParam'   : True,
           'chainSubsamples' : 150}

tic = time.time()

# Unpack the argument dictionary and call the collector.
myCollector = PolymerPy.WLCCollector(**simArgs)
toc = time.time()

print('Total simulation time: %f' % (toc - tic))
