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

def loadModel(dbName):
    """Loads a model polymer by reading the database generated from
    the polymer simulation.

    Parameters
    ----------
    

    """
    myDB = NPDB.NumPyDB_pickle(dbName, mode = 'load')

    mapName = ''.join([dbName, '.map'])
    with open(mapName, 'r') as file:
        fileLength = sum(1 for _ in file)
        file.seek(0) # Rewind file to the beginning

        simResults = {}
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

def computeLLH(dbName, dataFName):
    """Computes the log-likelihood of all simulated parameters.

    computeLLH computes the log-likelihood for all the simulated
    parameter values in the simulation. It is a wrapper around a few
    other functions and is intended to be an ease-of-use utility for
    processing full experiments.

    """
    simResults = loadModel(dbName)
    data = np.loadtxt(dataFName)

    # Initialize numpy array for holding parameter-pair and LLH values
    llhLandscape = np.zeros((len(simResults),), dtype=('f4,f4,f4'))
    
    for ctr, key in enumerate(simResults):
        #llh = computeSingleLLH(simResults[key], data)
        llh = computeSingleLLH_KDE(simResults[key], data)
        c, lp = key

        # llhLandscape is a structured numpy array.
        llhLandscape[ctr] = (c, lp, llh)

    return llhLandscape

def computeSingleLLH_KDE(probFunc, data):
    """Compute the log-likelihood of a given dataset use KDE.

    computeSingleLLH_KDE determines the log-likelihood of a dataset
    given the full simulated data. It uses kernal density estimation
    to approximate the continuous probability distribution function
    from the simulations.

    Parameters
    ----------
    probFunc : tuple of numpy arrays of floats
        A tuple at least 4 elements, the last being the complete
        simulated dataset..
    data : array of floats
        The data array.

    Returns
    -------
    llh : float
        The log-likelihood for the dataset.

    """
    # Add an axis so scikit-learn functions can operate on the data
    #Rg = probFunc[3][:, np.newaxis]
    # This works on the bumped data, not the polymer ground truth Rg.
    Rg = probFunc[4][:, np.newaxis]
    data = data[:, np.newaxis]

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

def computeSingleLLH(probFunc, data):
    """Compute the log-likelihood of a given dataset.

    computeSingleLLH determines the log-likelihood of a dataset
    given a probability mass function for an experiment.

    Parameters
    ----------
    probFunc : tuple of numpy arrays of floats

        A tuple of two numpy arrays. One array represents the
        normalized probability assigned to a bin and the other
        array represents the bin edges. The bin array must be one
        element larger than the probability array.
    data : array of floats
        The data array.

    Returns
    -------
    llh : float
        The log-likelihood for the dataset.
    """
    try:
        if len(probFunc[0]) == len(probFunc[1]) + 1:
            bins = probFunc[0]
            prob = probFunc[1]
        elif len(probFunc[1]) == len(probFunc[0]) + 1:
            bins = probFunc[1]
            prob = probFunc[0]
        else:
            raise SizeException

    except TypeError:
        print('TypeError')
        print('probFunc must be a tuple containing two arrays.')

    except SizeException:
        errorStr = dedent('''
            SizeException
            Length of probFunc[0]: %r
            Length of probFunc[1]: %r

            One of these values must be one greater than the
            other.
            '''
            % (len(probFunc[0]), len(probFunc[1])))

        print(errorStr)

    inds = np.digitize(data, bins)

    # Find the probability associated with each data point given
    # the input probability distribution/mass function
    numDataPoints = len(data)
    probPerPoint = [sortLLH(data[ctr], inds[ctr], len(bins), prob) \
                    for ctr in range(numDataPoints)]

    # Remove zeros from the returned array
    probabilities = np.fromiter(probPerPoint, np.float)
    #probabilities = probabilities[np.flatnonzero(probabilities)]

    try:
        llh = np.sum(np.log(probabilities))
    except: 
        sys.exit('0 found in probabilities.')

    return llh


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
    dbName = 'rw_2014-12-22'

    llh = computeLLH(dbName, dataFName)

    import matplotlib.pyplot as plt

    # Unpack the data structured array
    c = llh['f0']
    lp = llh['f1']

    # Reshape the variables to make a square grid
    c = np.unique(c)
    lp = np.unique(lp)
    C, LP = np.meshgrid(c,lp)

    L = np.sort(llh, order='f1').reshape(C.shape)
    LLH = L['f2']

    isolevels = -np.logspace(6, 4, 15)
    plt.figure()
    CS = plt.contour(c, lp, LLH, levels = isolevels)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Parameter space')
    plt.xlabel('Packing density, bp/nm')
    plt.ylabel('Persistence length, nm')

    
