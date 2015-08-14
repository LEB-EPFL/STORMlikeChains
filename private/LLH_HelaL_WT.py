"""Compute the log-likelihood for the wildtype Hela L telomeres.

The measured example data is in example_Measured_Dist.txt.

"""

__author__ = 'Kyle M. Douglass'
__email__  = 'kyle.douglass@epfl.ch'

# Ensure the script can access PolymerPy during development.
# Add PolymerPy path to PYTHONPATH during installation.
import sys
if len(sys.argv) > 1:
    if sys.argv[1] == '-d':
        sys.path.append('/home/douglass/src/PolymerPy/')

# Example begins here
from numpy import save
from PolymerPy import PolymerPy_helpers as Helpers

import os
os.chdir('..')

experimentDatasetName = os.getcwd() + '/experimental_distrs/Original_Data_L_dataset_RgTrans'

# List of NumPyDB database names without the file suffix.
dbNames = [os.getcwd() + '/simulation_data/rw_2015-1-26_HelaL_WT']

outputFName = 'likelihood_data/llh_Original_Data_L_dataset_RgTrans2015-1-26.npy'

# Compute the log-likelihood for each simulated distribution.
llh = Helpers.computeLLH(dbNames, experimentDatasetName)

# Save the log-likelihood data to a .npy file.
with open(outputFName, mode = 'wb') as fileOut:
    save(fileOut, llh)