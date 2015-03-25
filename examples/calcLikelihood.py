"""PolymerPy Example: Compute log-likelihood data.

This example demonstrates how to take the simulated data in
example_WLC_DB.dat and compute the likelihood that a simulated
distribution of radius of gyration values led to a measured dataset.

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

experimentDatasetName = 'example_Measured_Dist.txt'

# List of NumPyDB database names without the file suffix.
dbNames = ['example_WLC_DB']


outputFName = 'example_LLH_Data.npy'

# Compute the log-likelihood for each simulated distribution.
llh = Helpers.computeLLH(dbNames, experimentDatasetName)

# Save the log-likelihood data to a .npy file.
with open(outputFName, mode = 'wb') as fileOut:
    save(fileOut, llh)




