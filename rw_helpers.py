"""Helper functions for analyzing random walk polymer data.

"""

__author__ = 'Kyle M. Douglass'
__version__ = '0.1'
__email__ = 'kyle.douglass@epfl.ch'

import numpy as np
import NumPyDB as NPDB
import sys
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV

np.seterr(divide='raise')

def computeRg(path):
    """Compute the radius of gyration of a path.

    computeRg() calculates the radius of gyration of a Path
    object. The Rg is returned as a single number.

    Parameters
    ----------
    path : Array fo floats
        This is a random walk path. The radius of gyration is computed
        from the endpoints of its individual segments.

    Returns
    -------
    Rg : float
        The radius of gyration of the path object.

    """
    secondMoments = np.var(path, axis = 0)
    Rg = (np.sum(secondMoments)) ** (0.5)

    return Rg

def WLCRg(c, Lp, N):

    """Return the theoretical value for the gyration radius.

    Parameters
    ----------
    c : float
        The linear density of base pairs in the chain.
    Lp : float
        The persistence length of the wormlike chain.
    N : float
        The number of base pairs in the chain.

    Returns
    -------
    meanRg : float 
       The mean gyration radius of a theoretical wormlike chain.
    """

    Rg2 = (Lp * N / c) / 3 - \
             Lp ** 2 + \
             2 * Lp ** 3 / (N / c) ** 2 * \
             ((N / c) - Lp * (1 - np.exp(- (N / c)/ Lp)))

    meanRg = Rg2 ** 0.5

    return meanRg


def bumpPoints(path, locPrecision):
        """Bumps the points in a random direction in 3D.

        Parameters
        ----------
        locPrecision : float
            The localization precision of the measurement. This is the
            standard deviation of the Gaussian distribution
            determining the bump distances.

        Returns
        -------
        bumpedPath : Path object instance
            A shallow copy of this Path instance, but with the x-,y-,
            and z-coordinates of the path field bumped in random
            directions.

        """
        rows, cols = path.shape
        bumpedPath = locPrecision * np.random.randn(rows, cols) + path

        return bumpedPath

def loadModel(dbNameList):
    """Loads a model polymer by reading the database generated from
    the polymer simulation.

    Parameters
    ----------
    dbNameList : list of string
        Name(s) of the NumPyDB object that contains the pickled data.

    Returns
    -------
    simResults : dictionary
        The results of a simulation in dictionary format. Keys denote
        the simulation parameters. Two arrays of gyration radii belong
        to each key; the first is for the unbumped data and the second
        is for the bumped data.

    """
    simResults = {}
    for dbName in dbNameList:
        myDB = NPDB.NumPyDB_pickle(dbName, mode = 'load')
        mapName = ''.join([dbName, '.map'])
        
        with open(mapName, 'r') as file:
            fileLength = sum(1 for _ in file)
            file.seek(0) # Rewind file to the beginning

            for line in file:
                # Isloate the parameters in each line of the .map file.
                # Requires strings formatted like 'c=X, lp=Y'.
                paramStr = line[line.find('c='):-1]
                firstComma = paramStr.find(',')

                model = myDB.load(paramStr)[0]

                c = float(paramStr[2:firstComma])
                lp = float(paramStr[firstComma + 5:])

                simResults[(c, lp)] = model

    return simResults

def computeLLH(dbName, dataFName, bump = True):
    """Computes the log-likelihood of all simulated parameters.

    computeLLH computes the log-likelihood for all the simulated
    parameter values in the simulation. It is a wrapper around a few
    other functions and is intended to be an ease-of-use utility for
    processing full experiments.

    Parameters
    ----------
    dbName : string
        Name of the database to load the simulated ata from.
    dataFName : string
        Name of file containing the radii of gyration data.
    bump : boolean (optional)
        Use the bumped or unbumped data in the ML reconstruction?
        (Default is True)
    """
    simResults = loadModel(dbName)
    data = np.loadtxt(dataFName)

    # Initialize numpy array for holding parameter-pair and LLH values
    llhLandscape = np.zeros((len(simResults),), dtype=('f4,f4,f4'))
    
    for ctr, key in enumerate(simResults):
        # Use bumped data or unbumped data?
        if bump:
            simRg = simResults[key][1]
        else:
            simRg = simResults[key][0]
            
        llh = computeSingleLLH_KDE(simRg, data)
        c, lp = key

        # llhLandscape is a structured numpy array.
        llhLandscape[ctr] = (c, lp, llh)

    return llhLandscape

def computeSingleLLH_KDE(simData, expData):
    """Compute the log-likelihood of a given dataset use KDE.

    computeSingleLLH_KDE determines the log-likelihood of a dataset
    given the full simulated data. It uses kernal density estimation
    to approximate the continuous probability distribution function
    from the simulations.

    Parameters
    ----------
    simData : numpy array of floats
        The simulated data array. This should be a one-dimensional
        array.
    expData : array of floats
        The experimental data array.

    Returns
    -------
    llh : float
        The log-likelihood for the dataset.

    """
    # Add an axis so scikit-learn functions can operate on the data
    Rg = simData[:, np.newaxis]
    data = expData[:, np.newaxis]

    # The following is used to estimate the bandwidth, but it's very
    # slow.
    """grid = GridSearchCV(KernelDensity(),
                        {'bandwidth' : np.linspace(1, 5, 25)},
                        cv = 20)
    grid.fit(Rg)
    print('Best bandwidth: %r' % grid.best_params_)"""

    #kde = KernelDensity(kernel = 'gaussian', bandwidth = grid.best_params_['bandwidth']).fit(Rg)
    kde = KernelDensity(kernel = 'gaussian', bandwidth = 2).fit(Rg)
    log_dens = kde.score_samples(data)

    return sum(log_dens)

def sortLLH(dataPoint, index, binLength, hist):
    """Helper function for sorting the probabilities by bins.

    """
    if index == 0 or index == binLength:
        # This can't be set to zero because log-likelihoods are used.
        probability = 1e-5
    else:
        if hist[index - 1] == 0:
            probability = 1e-5
        else:
            probability = hist[index - 1]
        
    return probability

if __name__ == '__main__':
    dataFName = 'saved_distrs/Original_Data_L_dataset_RgTrans.txt'
    dbNames = ['rw_2015-1-14_HelaL_WT',
               'rw_2015-1-15_HelaL_WT',
               'rw_2015-1-16_HelaL_WT']

    llh = computeLLH(dbNames, dataFName)

    import matplotlib.pyplot as plt
    from scipy import interpolate

    # Unpack the data structured array
    c = llh['f0']
    lp = llh['f1']

    # Make a square grid for plotting the likelihood function
    cSpace = 1 # bp/nm
    lpSpace = 1 # nm
    cRange = np.arange(min(c), max(c) + cSpace, cSpace)
    lpRange = np.arange(min(lp), max(lp) + lpSpace, lpSpace)
    C, LP = np.meshgrid(cRange,lpRange)

    # Interpolate the likelihood function onto the generated grid
    rbf = interpolate.Rbf(c, lp, llh['f2'], function = 'linear')
    LLH = rbf(C, LP)

    """isolevels = -np.logspace(6, 4, 15)
    plt.figure()
    CS = plt.contour(C, LP, LLH, levels = isolevels)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.scatter(C.flatten(), LP.flatten())
    plt.title('Parameter space')
    plt.xlabel('Packing density, bp/nm')
    plt.ylabel('Persistence length, nm')
    plt.show()"""

plt.imshow(LLH, vmin = llh['f2'].min(), vmax = llh['f2'].max(),
           origin = 'lower',
           extent=[c.min(), c.max(), lp.min(), lp.max()],
           aspect = 'auto')
plt.scatter(c, lp, c = llh['f2'])
plt.colorbar()
plt.xlim((c.min(), c.max()))
plt.ylim((lp.min(), lp.max()))
plt.show()
