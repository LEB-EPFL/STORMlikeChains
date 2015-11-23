Manuscript Materials
====================

The contents of this folder contain data and scripts for reproducing
the simulations in the manuscript.

## Contents ##
### Files ###
+ *README.md* - This help file.
+ *HeLaLGenomicLengths.txt* - Genomic lengths in units of kilobase
pairs randomly drawn from the HeLa L genomic length probability
distribution.
+ *HeLaSGenomicLengths.txt* - Genomic lengths in units of kilobase
pairs randomly drawn from the HeLa S genomic length probability
distribution.
+ *Original_Data_L_dataset_RgTrans* - Experimental radii of gyration
from HeLa L telomeres.
+ *Original_Data_S_dataset_RgTrans* - Experimental radii of gyration
from HeLa S telomeres.
+ *simulate_HeLaL.py* - Python script to simulate HeLa L wormlike
chains with the same polymer parameters and measurement parameters as
noted in the manuscript.
+ *simulate_HeLaS.py* - Python script to simulate HeLa S wormlike
chains with the same polymer parameters and measurement parameters as
noted in the manuscript.

## Instructions ##

Once your Python 3 environment is setup correctly, you may run the
scripts *simulate_HeLaL.py* and *simulateHeLaS.py* to generate
simulated datasets. These datasets may be compared to the experimental
radius of gyration data found in *Original_Data_L_dataset_RgTrans* and
*Original_Data_S_dataset_RgTrans*, respectively, using the helper
functions as described in the _examples_ directory.

