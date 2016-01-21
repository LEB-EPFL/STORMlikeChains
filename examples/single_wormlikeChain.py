"""STORMlikeChains Example: Simulate an ensemble of wormlike chains.

This example script shows how to setup the the wormlike chain
collector and run a simulation. The collector object takes a single
pair of parameter values and generates wormlike chain ensembles. The
radius of gyration of each chain conformation in the ensemble is saved
in a NumPy database (NumPyDB). This is simply a class that wraps
around Python's CPickle utility.

Additionally, a single molecule localization experiment is simulated
by specifying a non-zero localization precision. This takes each
monomer in a chain conformation and displaces it randomly in each
direction according to a Gaussian distribution whose standard
deviation is equivalent to the localization precision.

"""

__author__ = 'Kyle M. Douglass'
__email__  = 'kyle.douglass@epfl.ch'

# Ensure the script can access STORMlikeChains during development.
# Add STORMlikeChains path to PYTHONPATH during installation.
import sys
if len(sys.argv) > 1:
    if sys.argv[1] == '-d':
        sys.path.append('/home/douglass/src/STORMlikeChains/')

# The example begins here.
from STORMlikeChains import STORMlikeChains

from numpy import ones, append, arange, array, concatenate, meshgrid
from numpy.random import random

import time

# Packing ratio and persistence length
C = array([40])
LP = array([50])

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

    This will usually be set to True, so all one must specify every
    pair of values that one wants to simulate.
chainSubsamples : int (optional)
    The number of segments in the chain to keep; models the small
    number of localizations obtained per chain. (Default is -1, keep
    all segments.)


"""
# Create a random numbers for the number of base pairs in each chain.
numPaths = 1000
# basePairDist = 24000 * (random(numPaths) - 0.5) + 27000
basePairDist = 33000 * ones(numPaths)

simArgs = {'numPaths'      : numPaths,
           'pathLength'    : basePairDist,
           'linDensity'    : C,
           'persisLength'  : LP,
           'segConvFactor' : 1, #2.5
           'nameDB'        : 'example_single_WLC_DB',
           'locPrecision'  : 15,
           'fullSpecParam' : True,
           'chainSubsamples' : 150} #150

tic = time.time()

# Unpack the argument dictionary and call the collector.
myCollector = STORMlikeChains.WLCCollector(**simArgs)
toc = time.time()

print('Total simulation time: %f' % (toc - tic))

from STORMlikeChains import STORMlikeChains_helpers
from numpy import sqrt

theoryRg3D = STORMlikeChains_helpers.WLCRg(40, 50, 33000)
theoryRg2D = theoryRg3D * sqrt(2/3)

print('Theoretical 3DtheoryRg: {0:.4f}'.format(theoryRg3D))
print('Theoretical 2DtheoryRg: {0:.4f}'.format(theoryRg2D))
