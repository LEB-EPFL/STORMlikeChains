"""Classes for simulating random walks in three dimensions.

"""

__author__ = 'Kyle M. Douglass'
__version__ = '0.2'
__email__ = 'kyle.douglass@epfl.ch'

from math import modf
from textwrap import dedent
from numpy import pi, cos, sin, arccos, meshgrid, sum, var, zeros
from numpy import array, cross, concatenate, hstack, cumsum, flatnonzero
from numpy import histogram, exp, mean, ceil, arange, digitize, log
from numpy import fromiter
from numpy import float as npFloat
from numpy.random import randn, random
from scipy.linalg import norm
import NumPyDB as NPDB
from datetime import datetime
import time

from scipy.linalg import get_blas_funcs
# Import nrm2 from FortranBLAS library optimized for vectors.
# I've found that this can be faster than scipy.linalg.norm().
nrm2, = get_blas_funcs(('nrm2',), dtype = 'float64')

# Find current date for naming the database.
currentTime = datetime.now()
year = currentTime.year
month = currentTime.month
day = currentTime.day
dateStr = "%s-%s-%s" % (year, month, day)

class Path():             
    def _randPointSphere(self, numPoints):
        """Randomly select points from the surface of a sphere.
    
        Parameters
        ----------
        numPoints : int
        The number of points to return.
    
        Returns
        -------
        points : array of float
            The x, y, and z coordinates of each point on the sphere.

        References
        ----------
        [1] Weisstein, Eric W. "Sphere Point Picking." From
        MathWorld--A Wolfram Web
        Resource. http://mathworld.wolfram.com/SpherePointPicking.html

        """
        uRand1 = random(numPoints)
        uRand2 = random(numPoints)
    
        # Random points on the unit sphere in spherical coordinates
        # phi is the azimuth angle and theta is the zenith angle
        phi = 2 * pi * uRand1
        theta = arccos(2 * uRand2 - 1)
    
        # Convert to Cartesian coordinates
        x = cos(phi) * sin(theta)
        y = sin(phi) * sin(theta)
        z = cos(theta)
        points = array([x, y, z])

        return points

    def _checkPath(self):
        """Checks that a path has the correct number of columns.

        """
        pathShape = self.path.shape
        if pathShape[1] != 2 and pathShape[1] != 3:
            errorStr = dedent('''
            Error: Path array has %d columns.
            A path must have either 2 or 3 columns.
            For 2D walks, the columns are the x and y coordinates.
            For 3D walks, the columns are the x, y, and z coordinates.
            ''' % pathShape[1])
            
            raise SizeException(errorStr)

class WormlikeChain(Path):
    """Creates a 3D wormlike chain.

    Parameters
    ----------
    numSegments : float
        The number of segments in the chain. Must be >= 1 and need not
        be an integer.
    pLength : int
        The persistence length in units of chain segments.
    initPoint : array of float, optional
        The coordiantes of the first point of the chain.

    Attributes
    ----------
    path : array of float
        2D array of floats whose columns describe the endpoints of the
        segments comprising the path.
        
        """
    def __init__(self,
                 numSegments,
                 pLength,
                 initPoint = array([1, 0, 0])):

        self.numSegments = numSegments
        self.pLength = pLength
        self.makeNewPath(initPoint)

    def _makePath(self, initPoint = array([1, 0, 0])):
        """Create the wormlike chain.

        The wormlike chain is created by first choosing the sizes of
        the small, random displacements in a plane tangent to a point
        on the surface of the unit sphere defined by the vector
        currPoint. The distribution for the sizes is given by the
        Boltzmann statistics for a semiflexible rod bending by a given
        angle due to interaction with its thermal environment.

        A random direction in this tangent plane is chosen by randomly
        and uniformly generating a vector on the unit sphere, taking
        its cross product with the currPoint vector, and normalizing
        the cross product. This cross product is multiplied by the
        size of the displacement found previously to generate the
        displacement vector.

        After displacing the currPoint vector into the tangent plane,
        the point in the plane is back projected onto the unit sphere
        to find the vector representing the next step in the polymer
        walk.

        This process is repeated until a number of vectors determined
        by numSegments representing a random walk on the surface of a
        sphere are generated. These vectors are cumulatively summed at
        the end to produce the final path field, which is the
        trajectory of the polymer.

        Parameters
        ----------
        initPoint : array of float
            Initial point to start polymer from on the unit sphere.

        """
        if self.numSegments < 1:
            errorStr = dedent('''
                The number of segments must be greater than 1, but a
                value of %r was supplied.''' % self.numSegments)
           
            raise ValueError(errorStr)
        
        numSegFrac, numSegInt = modf(self.numSegments)
        numSegInt = int(numSegInt)
        
        # Create the displacement distances in the tangent planes
        angDisp = (2 / self.pLength) ** (0.5) \
                   * randn(numSegInt - 1)
        tanPlaneDisp = sin(angDisp)

        # Create random vectors uniformly sampled from the unit sphere
        randVecs = self._randPointSphere(numSegInt - 1)
        
        # Final small displacement for non-integer numSegments
        if numSegFrac != 0.0:
            angDispFinal = (self.pLength * numSegFrac) ** (-0.5) \
                           * randn(1)
            tanPlaneDispFinal = numSegFrac * sin(angDispFinal)
            randVecFinal = numSegFrac * self._randPointSphere(1)

            # Append final direction vector and displacements
            angDisp = concatenate((angDisp, angDispFinal))
            tanPlaneDisp = concatenate((tanPlaneDisp,
                                        tanPlaneDispFinal))
            randVecs = hstack((randVecs, randVecFinal))

        # Primary iterative loop for creating the chain
        currPoint = initPoint
        workingPath = zeros((numSegInt + 1 , 3))
        workingPath[0, :] = currPoint
        for ctr in range(len(tanPlaneDisp)):
            # Create a displacement in the plane tangent to currPoint
            crossX = ((currPoint[1] * randVecs[2,ctr]) - \
                       (currPoint[2] * randVecs[1,ctr]))
            crossY = ((currPoint[2] * randVecs[0,ctr]) - \
                       (currPoint[0] * randVecs[2,ctr]))
            crossZ = ((currPoint[0] * randVecs[1,ctr]) - \
                       (currPoint[1] * randVecs[0,ctr]))
            dispVector = array([crossX, crossY, crossZ])

            # Check if displacement and currPoint vectors are parallel
            while nrm2(dispVector) == 0:
                newRandVec = self._randPointSphere(1)
                dispVector = cross(currPoint, newRandVec)

            # Move the currPoint vector in the tangent plane
            # (I seem to get faster norms when calling the BLAS
            # function from this point instead of scipy.linalg.norm.)
            dispVector = (dispVector / nrm2(dispVector)) \
              * tanPlaneDisp[ctr]
            
            # Back project new point onto sphere
            projDistance = 1 - cos(angDisp[ctr])
            nextPoint = (1 - projDistance) * currPoint + dispVector
                        
            # Append nextPoint to array of points on the path
            workingPath[ctr + 1, :] = nextPoint
            currPoint = nextPoint

        # Add up the vectors in path to create the polymer
        self.path = cumsum(workingPath, axis = 0)

    def computeRg(self):
        """Compute the radius of gyration of a path.

        computeRg() calculates the radius of gyration of a Path object
        and assigns it to a field within the same Path object.

        """
        pathShape = self.path.shape
        if pathShape[1] != 2 and pathShape[1] != 3:
            errorStr = dedent('''
            Error: Path array has %d columns.
            A path must have either 2 or 3 columns.
            For 2D walks, the columns are the x and y coordinates.
            For 3D walks, the columns are the x, y, and z coordinates.
            ''' % pathShape[1])
            
            raise SizeException(errorStr)

        secondMoments = var(self.path, axis = 0)
        Rg = (sum(secondMoments)) ** (0.5)

        return Rg

    def makeNewPath(self, initPoint = array([1, 0, 0])):
        """Clears current path and makes a new one.

        Parameters
        ----------
        initPoint : array of float
            Coordinates of the first point.

        """
        self.path = initPoint
        self._makePath()

        # Ensure the path field will work with other objects.
        self._checkPath()

class Analyzer():
    """Analyzes histograms of polymer paths.

    Parameters
    ----------
    dbName : str
        The name of the pickle database that contains the histogram
        data.

    Attributes
    ----------
    myDB : NumPyDB_pickle object
        Object for writing and loading radius of gyration histograms.
        
    """
    def __init__(self, dbName):
        self.dbName = dbName
        self._myDB = NPDB.NumPyDB_pickle(dbName, mode = 'load')

    def computeMeanRg(self, identifier):
        """Compute the mean radius of gyration from histogram data.

        Parameters
        ----------
        identifier : str
            String identifier for which dataset to import.

        Returns
        -------
        meanRg : float
            The mean radius of gyration determined from the histogram.

        """
        importData = self._myDB.load(identifier)
        myHist = importData[0][0]
        myBins = importData[0][1]
        binWidth = importData[0][2]

        # Find the centers of each histogram bin
        binCenters = myBins + binWidth / 2
        binCenters = binCenters[0:-1]

        meanRg = sum(binCenters * myHist * binWidth)
        return meanRg

    def computeLLH(self, probFunc, data):
        """Compute the log-likelihood of a given dataset.

        computeLLH(self) determines the log-likelihood of a dataset
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
                Length of probFunc[0]: %r
                Length of probFunc[1]: %r

                One of these values must be one greater than the
                other.
                '''
                % (len(probFunc[0]), len(probFunc[1])))

            print('SizeException')
            print(errorStr)
            
        except:
            errorStr = dedent('''
            Unexpected error occurred. The arguments to computeLLH()
            may be of incorrect type.''')
            print(errorStr)

        inds = digitize(data, bins)

        # Find the probability associated with each data point given
        # the input probability distribution/mass function
        numDataPoints = len(data)
        probPerPoint = [self._sortLLH(data[ctr],
                                     inds[ctr],
                                     len(bins),
                                     prob) for ctr in range(numDataPoints)]

        # Remove zeros from the returned array
        probabilities = fromiter(probPerPoint, npFloat)
        probabilities = probabilities[flatnonzero(probabilities)]
        
        print(log(probabilities))
        llh = sum(log(probabilities))

        return llh

    def _sortLLH(self, dataPoint, index, binLength, hist):
        """Helper function for sorting the probabilities by bins.
        
        """
        if index == 0 or index == binLength:
            probability = 0
        else:
            probability = hist[index -1]

        return probability

    def WLCRg(self, c, Lp, N):

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
                 ((N / c) - Lp * (1 - exp(- (N / c)/ Lp)))

        meanRg = Rg2 ** 0.5

        return meanRg

class Collector():
    """Creates random walk paths and collects their statistics.

    A Collector generates a user-defined number of random walk paths
    with possibly different lengths by sending the walk parameters to
    a Path object. After the path has been generated, the statistics
    that describe the path are collected and binned into a histogram.

    Parameters
    ----------
    numPaths : int
        The number of paths to collect before stopping the simulation
    pathLength : array of float
        The length of each simulated path
    segConvFactor : float
        Conversion factor between the user units and path segments
        (Default is 1)

    Attributes
    ----------
    myPath : path object
        The path for generating walks.
        
    """

    def __init__(self,
                 numPaths,
                 pathLength,
                 segConvFactor = 1,
                 nameDB = 'rw_' + dateStr):
        if numPaths != len(pathLength):
            errorStr = dedent('''
                Number of paths input: %r
                Length of path lengths vector: %r

                These values must be equal and integers.'''
                % (numPaths, len(pathLength)))
            
            raise SizeException(errorStr)
        
        self.numPaths = numPaths
        self._segConvFactor = segConvFactor
        self._nameDB = nameDB
        self.__pathLength = self._convSegments(pathLength, True)

    def _convSegments(self, pathParam, multiplyBool):
        """Convert path parameters into segments.

        Parameters
        ----------
        pathParam : array of float
            The parameters to convert into segments
        multiplyBool : bool
            Multiply or divide by the conversion factor

        Returns
        -------
        paramInSegments : array of float
            The parameters in units of path segments

        """
        if multiplyBool:
            paramInSegments = pathParam * self._segConvFactor
        else:
            paramInSegments = pathParam / self._segConvFactor
            
        return paramInSegments

class WLCCollector(Collector):
    """Collector for the wormlike chain.

    Parameters
    ----------
    numPaths : int
        The number of paths to collect before stopping the simulation
    pathLength : array of float
        The length of each simulated path in genomic length
    linDensity : float
        The number of base pairs per user-defined unit of length
    persisLength : float
        The path's persistence length in user-defined units of length
    segConvFactor : float (optional)
        Conversion factor between the user units and path segments
        (Default is 1)
    myAnalyzer : Analyzer
        The analyzer for computing the random walk statistics.
        (Default analyzer does not filter out any walks.)

    """
    def __init__(self,
                 numPaths,
                 pathLength,
                 linDensity,
                 persisLength,
                 segConvFactor = 1,
                 nameDB = 'rw_' + dateStr):
        super().__init__(numPaths, pathLength, segConvFactor, nameDB)
        self._linDensity = self._convSegments(linDensity, False)
        self._persisLength = self._convSegments(persisLength, True)
        self.__pathLength = pathLength

        self._startCollector()

    def _startCollector(self):
        """Begin collecting wormlike chain conformation statistics.

        """
        linDensity, persisLength = meshgrid(self._linDensity,
                                            self._persisLength)

        myDB = NPDB.NumPyDB_pickle(self._nameDB)
        
        # Loop over all combinations of density and persistence length
        for c, lp in zip(linDensity.flatten(),
                         persisLength.flatten()):

            numSegments = self.__pathLength / c
            
            # Does the collector already have a path object?
            if not hasattr(self, '_myPath'):
                self._myPath = WormlikeChain(numSegments[0], lp)

            # Main loop for creating paths
            Rg = zeros(numPaths)
            for ctr in range(self.numPaths):
                self._myPath.numSegments = numSegments[ctr]
                self._myPath.pLength = lp
                self._myPath.makeNewPath()

                # Analyze the new path for its statistics
                #if hasattr(self, '_myAnalyzer'):
                #    currRg = self._myAnalyzer.computeRg(self._myPath)
                #    Rg[ctr] = currRg
                Rg[ctr] = self._myPath.computeRg()

            """=======================================================
            Possibly move everything below here to a function for
            organizational purposes and clarity.
            """
                
            # Convert back to user-defined units
            c = self._convSegments(c, True)
            lp = self._convSegments(lp, False)
            Rg = self._convSegments(Rg, False)

            print('Density: %r, Persistence length: %r'
                  %(c, lp))

            try:
                # Create histogram of Rg data
                stdRg = var(Rg) ** (0.5)
                numRg = len(Rg)
                '''The following is from Scott, Biometrika 66, 605 (1979)

                '''
                binWidth = 3.49 * stdRg * numRg ** (-1/3)
                numBins = ceil((max(Rg) - min(Rg)) / binWidth)
                hist, bin_edges = histogram(Rg, numBins, density = True)
            
                # Save the gyration radii histogram to the database
                identifier = 'c=%s, lp=%s' % (c, lp)
                myDB.dump((hist, bin_edges, binWidth), identifier)
                print('Mean of all path Rg\'s: %f' % mean(Rg))
            except:
                pass

class SizeException(Exception):
    pass
            
if __name__ == '__main__':
    # Test case 1: test sphere sampling
    """import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    points = randPointSphere(500)

   # norms = (points[0,:] ** 2 + points[1,:] ** 2 + points[2,:] ** 2)
   # print(norms)
        
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(points[0,:], points[1,:], points[2,:])
    plt.show()"""

    # Test case 2: test for vector normalization
    """myChain = WormlikeChain(100, 25)
    vectors = myChain._randPointSphere(10)

    # Create vectors of random lengths
    vectors = vectors * random(vectors.shape)
    nVectors = myChain._normVector(vectors)

    print('Vectors: %r' % vectors)
    print('Norm of the vectors: %r' % norm(vectors, axis = 0))
    print('Normalized vectors: %r' % nVectors)
    print('Norm of the normalized vectors: %r'
          % norm(nVectors, axis = 0))
    """
        
    # Test case 3: Create a single random walk
    """import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    numSegments, pLength = 1000.1, 25
    myChain = WormlikeChain(numSegments, pLength)
    path = myChain.path
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(path[:,0], path[:,1], path[:,2])
    plt.show()"""

    # Test case 4: Create wormlike chain collector.
    """numPaths = 5 # Number of paths per pair of walk parameters
    pathLength = (2000) **(0.5) * randn(numPaths) + 25000 # bp in walk
    linDensity = array([40, 50]) # bp / nm
    persisLength = array([15, 20, 25]) # nm
    segConvFactor = 10 / min(persisLength) # segments / min persisLen

    myCollector = WLCCollector(numPaths,
                               pathLength,
                               linDensity,
                               persisLength,
                               segConvFactor)"""

    # Test case 5: Compute the Rg of a WLC. 
    """numPaths = 5 # Number of paths per pair of walk parameters
    pathLength = (2000) **(0.5) * randn(numPaths) + 25000 # bp in walk
    linDensity = array([40, 50]) # bp / nm
    persisLength = array([15, 20, 25]) # nm
    segConvFactor = 10 / min(persisLength) # segments / min persisLen

    myCollector = WLCCollector(numPaths,
                               pathLength,
                               linDensity,
                               persisLength,
                               segConvFactor)"""

    # Test case 6: Test whether the computed Rg matches theory.
    """from numpy import ones
    numPaths = 1000 # Number of paths per pair of walk parameters
    pathLength =  25000 * ones(numPaths) # bp in walk
    linDensity = array([100]) # bp / nm
    persisLength = array([100]) # nm
    segConvFactor = 25 / min(persisLength) # segments / min persisLen
    nameDB = 'rw_' + dateStr

    myCollector = WLCCollector(numPaths,
                               pathLength,
                               linDensity,
                               persisLength,
                               segConvFactor,
                               nameDB)

    myAnalyzer = Analyzer(nameDB)
    identifier = 'c=%0.1f, lp=%0.1f' % (linDensity, persisLength)
    meanSimRg = myAnalyzer.computeMeanRg(identifier)
    meanTheorRg = myAnalyzer.WLCRg(linDensity, persisLength, pathLength[0])
    print(dedent('''
                 The mean of the simulated distribution is %f.
                 The mean theoretical gyration radius is %f.'''
                 % (meanSimRg, meanTheorRg)))"""

    # Test case 7: Test the computed Rg's over a range of parameters
    from numpy import ones, append
    import matplotlib.pyplot as plt
    numPaths = 50000 # Number of paths per pair of walk parameters
    pathLength =  16000 * (random(numPaths) - 0.5) + 25000 # bp in walk
    linDensity = arange(10, 110, 20)  # bp / nm
    persisLength = arange(10, 210, 20) # nm
    segConvFactor = 25 / min(persisLength) # segments / min persisLen
    nameDB = 'rw_' + dateStr

    tic = time.clock()
    myCollector = WLCCollector(numPaths,

                               pathLength,
                               linDensity,
                               persisLength,
                               segConvFactor,
                               nameDB)
    toc = time.clock()
    print('Total processing time: %f' % (toc - tic))
    
    myAnalyzer = Analyzer(nameDB)

    """c, lp = meshgrid(linDensity, persisLength)
    errorRg = array([])
    
    # Loop over all combinations of density and persistence length
    for cCtr, lpCtr in zip(c.flatten(), lp.flatten()):

        identifier = 'c=%0.1f, lp=%0.1f' % (cCtr, lpCtr)
        meanSimRg = myAnalyzer.computeMeanRg(identifier)
        meanTheorRg = myAnalyzer.WLCRg(cCtr, lpCtr, pathLength[0])
        errorRg = append(errorRg,
                          abs(meanSimRg - meanTheorRg) / meanTheorRg)
        
        print(dedent('''
                     c=%0.1f, lp=%0.1f
                     The mean of the simulated distribution is %f.
                     The mean theoretical gyration radius is %f.
                     The error in the mean is %f.'''
                     % (cCtr,
                        lpCtr,
                        meanSimRg,
                        meanTheorRg,
                        abs(meanSimRg - meanTheorRg) / meanTheorRg)))

    plt.hist(errorRg)
    plt.xlabel(r'Percent error in mean $R_g$ values')
    plt.ylabel('Number of occurrences')
    plt.grid(True)
    plt.show()"""

    # Test case 8: Profile the WormlikeChain
    """from numpy import ones, append
    numPaths = 1 # Number of paths per pair of walk parameters
    pathLength =  25000 * ones(numPaths) # bp in walk
    linDensity = arange(20, 120, 20)  # bp / nm
    persisLength = arange(20, 220, 20) # nm
    segConvFactor = 25 / min(persisLength) # segments / min persisLen
    nameDB = 'rw_' + dateStr

    myCollector = WLCCollector(numPaths,
                               pathLength,
                               linDensity,
                               persisLength,
                               segConvFactor,
                               nameDB)"""

    # Test case 9: Testing the loglikelihood construction in Analyzer
    """nameDB = 'rw_2014-12-10'
    myAnalyzer = Analyzer(nameDB)
    
    #myAnalyzer.computeLLH() # Returns a TypeError
    #myAnalyzer.computeLLH(probFunc = [1], data = [1,2,3]) # Returns TypeError
    #myAnalyzer.computeLLH(probFunc = ([1], [2]), data = [1,2,3]) # SizeException
    
    llh = myAnalyzer.computeLLH(probFunc = ([0.25, 0.5, 0.25], [-1, 0, 1, 2]),
                                data = range(-2, 4))

    print(llh)"""
