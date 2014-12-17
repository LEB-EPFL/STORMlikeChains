"""Helper functions for analyzing random walk polymer data.

"""

__author__ = 'Kyle M. Douglass'
__version__ = '0.1'
__email__ = 'kyle.douglass@epfl.ch'

import numpy as np
import NumPyDB as NPDB


def loadData(fName):
    pass

def loadModel(dbName):
    """Loads a model polymer by reading the database generated from
    the polymer simulation.

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
    other functions and is intended to be an ease-of-use utility.

    """
    simResults = loadModel(simResults)
    data = loadData(dataFName) #WRITE THE LOAD DATA FUNCTION

    # Initialize numpy array for holding parameter-pair and LLH values
    llhLandscape = np.zeros((len(simResults), 2))
    
    for ctr, key in enumerate(simResults):
        llh = computeSingleLLH(simResults[key], data)
        llhLandscape[ctr,0] = key
        llhLandscape[ctr,1] = llh

    return llhLandscape

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
    probabilities = np.fromiter(probPerPoint, npFloat)
    probabilities = probabilities[np.flatnonzero(probabilities)]

    llh = sum(log(probabilities))
    return llh

def sortLLH(dataPoint, index, binLength, hist):
    """Helper function for sorting the probabilities by bins.

    """
    if index == 0 or index == binLength:
        probability = 0
    else:
        probability = hist[index - 1]

    return probability

if __name__ == '__main__':
    pass

