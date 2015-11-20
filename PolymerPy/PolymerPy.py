"""Classes for simulating random walk models for polymer physics and
DNA/chromatin studies with STORM microscopy.

"""

__author__ = 'Kyle M. Douglass'
__email__  = 'kyle.douglass@epfl.ch'

from math import modf
from textwrap import dedent
from numpy import pi, cos, sin, arccos, meshgrid, sum, var, zeros
from numpy import array, \
                  cross, \
                  concatenate, \
                  hstack, \
                  cumsum, \
                  flatnonzero
from numpy import histogram, exp, mean, ceil, arange, digitize, log
from numpy import fromiter, newaxis, linspace
from numpy import float as npFloat
from numpy.random import randn, random, choice
from numpy.linalg import norm
import PolymerPy.NumPyDB as NPDB
from datetime import datetime
import time
import multiprocessing
from PolymerPy.PolymerPy_helpers import computeRg, \
                                        bumpPoints, \
                                        WLCRg, \
                                        loadModel

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
        currPoint = initPoint / norm(initPoint)
        workingPath = zeros((numSegInt + 1 , 3))
        workingPath[0, :] = currPoint
        for ctr in range(len(tanPlaneDisp)):
            # Create a displacement in the plane tangent to currPoint
            # (Hard coding the cross product is faster than numpy's
            # implementation for only two vectors.)
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
        
    def makeNewPath(self, initPoint = array([1, 0, 0])):
        """Clears current path and makes a new one.

        Parameters
        ----------
        initPoint : array of float
            Coordinates of the first point.

        """        
        self.path = initPoint
        self._makePath(initPoint)

        # Ensure the path field will work with other objects.
        self._checkPath()        

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
    numPaths        : int
        The number of paths to collect before stopping the simulation
    pathLength      : array of float
        The length of each simulated path in genomic length
    linDensity      : float
        The number of base pairs per user-defined unit of length
    persisLength    : float
        The path's persistence length in user-defined units of length
    segConvFactor   : float (optional)
        Conversion factor between the user units and path segments
        (Default is 1)
    locPrecision    : float (optional)
        Standard deviation of the Gaussian defining the effective
        system PSF. (Default is 0, meaning no bumps are made)
    fullSpecParam   : bool (optional)
        Do linDensity and persisLength define all the parameter-space
        points to simulate, or do they instead define the points in a
        grid to be generated with meshgrid? In the first case, the
        number of points to simulate is equal to the length of
        persisLength OR linDensity, whereas in the second case it's
        equal to the number of points in persisLength TIMES the number
        of points in linDensity. (Default is false; the points will
        define a grid in this case).
    chainSubsamples : int (optional)
        The number of segments in the chain to keep; models the small number of
        localizations obtained per chain. (Default is -1, keep all segments.)
        
    """
    def __init__(self, **kwargs):

        # Unpack the arguments
        numPaths     = kwargs['numPaths']
        pathLength   = kwargs['pathLength']
        linDensity   = kwargs['linDensity']
        persisLength = kwargs['persisLength']

        if 'segConvFactor' in kwargs:
            segConvFactor = kwargs['segConvFactor']
        else:
            segConvFactor = 1

        if 'nameDB' in kwargs:
            nameDB = kwargs['nameDB']
        else:
            nameDB = 'rw_' + dateStr

        if 'locPrecision' in kwargs:
            locPrecision = kwargs['locPrecision']
        else:
            locPrecision = 0

        if 'fullSpecParam' in kwargs:
            self._fullSpecParam = kwargs['fullSpecParam']
        else:
            self._fullSpecParam = False
            
        if 'chainSubsamples' in kwargs:
            self._chainSubsamples = kwargs['chainSubsamples']
        else:
            self._chainSubsamples = -1
        
        super().__init__(numPaths, pathLength, segConvFactor, nameDB)

        # Convert from user-defined units to simulation units
        self._linDensity   = self._convSegments(linDensity, False)
        self._persisLength = self._convSegments(persisLength, True)
        self._locPrecision = self._convSegments(locPrecision, True)

        self.__pathLength = pathLength

        self._startCollector()
    
    def _startCollector(self):
        """Begin collecting wormlike chain conformation statistics.

        """
        if self._fullSpecParam:
            # Zip together the two arrays
            loopParams = list(zip(self._linDensity,
                                  self._persisLength))
        else:
            # Generate a grid of points and then zip it
            linDensity, persisLength = meshgrid(self._linDensity,
                                                self._persisLength)
            loopParams = list(zip(linDensity.flatten(),
                             persisLength.flatten()))
        
        myDB = NPDB.NumPyDB_pickle(self._nameDB)

        myChains = []
        # Create a list of chains, one for each parameter-pair value
        # Each chain will be run independently on different cores
        for c, lp in loopParams:

            # This is an array of (in general) different values
            numSegments = self.__pathLength / c
            
            # Create new WormlikeChain instance and add it to the list
            myChain = WormlikeChain(numSegments[0], lp)
            myChains.append({'chain'           : myChain,
                             'numSegments'     : numSegments,
                             'locPrecision'    : self._locPrecision,
                             'chainSubsamples' : self._chainSubsamples})

        # Compute the gyration radii for all the parameter pairs
        pool   = multiprocessing.Pool()
        RgData = pool.map(parSimChain, myChains)
        pool.close(); pool.join()

        # Unpack the gyration radii and save them to the database
        for ctr, (c, lp) in enumerate(loopParams):

            # Unpack the computed RgData
            currRgData = RgData[ctr]
            Rg         = currRgData['Rg']
            RgBump     = currRgData['RgBump']
        
            # Convert back to user-defined units
            c  = self._convSegments(c, True)
            lp = self._convSegments(lp, False)
            Rg = self._convSegments(Rg, False)
            if self._locPrecision != 0:
                RgBump = self._convSegments(RgBump, False)

            print('Density: %r, Persistence length: %r'
                  %(c, lp))

            try:
                # Save the gyration radii histogram to the database
                identifier = 'c=%s, lp=%s' % (c, lp)
                myDB.dump((Rg, RgBump), identifier)
                print('Mean of all path Rg\'s: %f' % mean(Rg))
            except:
                print('A problem occurred while saving the data.')

                
def parSimChain(data):
    """Primary processing for-loop to be parallelized.

    parSimChain(data) is the most intensive part of the simulation. It
    is a function applied to a WormlikeChain instance and repeatedly
    calculates new conformations and gyration radii for those
    conformations. Each WormlikeChain instance was defined with a
    different persistence length.

    Parameters
    ----------
    data : dictionary
        The data dictionary contains chain, numSegments and
        locPrecision keys. The chain is the WormlikeChain instance and
        the numSegments array contains the number of segments to
        simulate for each chain iteration. locPrecision is the
        localization precision used for bumping the chain locations.

    Returns
    -------
    RgDict : dictionary
        Dictionary with Rg and RgBump keys containing the gyration
        radii for the chain and its sampled version.
    """
    
    chain           = data['chain']
    numSegments     = data['numSegments']
    locPrecision    = data['locPrecision']
    chainSubsamples = data['chainSubsamples']

    numPaths = len(numSegments)
    
    Rg = zeros(numPaths)
    RgBump = zeros(numPaths)
    for ctr in range(numPaths):
        # Randomize the starting position
        randStartDir = random(3) - 0.5
        
        chain.numSegments = numSegments[ctr]
        chain.makeNewPath(initPoint = randStartDir)
        
        # Downsample the chain
        if (chainSubsamples == -1):
            # Keep all segments, i.e. don't downsample
            downsampledPath = chain.path
        else:
            try:
                allRowIndexes   = arange(0, chain.numSegments)
                keepTheseRows   = choice(allRowIndexes,
                                         chainSubsamples,
                                         replace = False)
                downsampledPath = chain.path[keepTheseRows.astype(int), :]
            except:
                print('Error in downsampling the chain. Keeping all segments.')
                print('Does the number of subsamples exceed the number of segments?')
                downsampledPath = chain.path
                
        Rg[ctr] = computeRg(downsampledPath, dimensions = 3)
        if locPrecision != 0:
            bumpedPath = bumpPoints(downsampledPath, locPrecision)
            RgBump[ctr] = computeRg(bumpedPath, dimensions = 3)

    RgDict = {'Rg' : Rg, 'RgBump' : RgBump}
    return RgDict
    
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

    # Test case 7: Test the computed Rg's over a range of parameters.
    # Used as main code for generating walks at the moment.
    """from numpy import ones, append
    import matplotlib.pyplot as plt
    numPaths = 250000 # Number of paths per pair of walk parameters
    pathLength =  5100 * (random(numPaths) - 0.5) + 11750 # bp in walk
    linDensity = arange(10, 110, 20)  # bp / nm
    persisLength = arange(10, 210, 20) # nm
    segConvFactor = 25 / min(persisLength) # segments / min persisLen
    nameDB = 'rw_' + dateStr
    locPrecision = 2.45 # nm

    tic = time.time()
    myCollector = WLCCollector(numPaths,
                               pathLength,
                               linDensity,
                               persisLength,
                               segConvFactor,
                               nameDB,
                               locPrecision)
    toc = time.time()
    print('Total processing time: %f' % (toc - tic))
    
    myAnalyzer = Analyzer(nameDB)

    c, lp = meshgrid(linDensity, persisLength)
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

    # Test case 10: Test KDE estimation
    # Perform a kernel density estimation on the data
    """import matplotlib.pyplot as plt
    from numpy import ones, append, linspace, min, max, newaxis
    
    numPaths = 100 # Number of paths per pair of walk parameters
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

    myDB = NPDB.NumPyDB_pickle(nameDB, mode = 'load')
    importData = myDB.load('c=100.0, lp=100.0')
    myHist = importData[0][0]
    myBins = importData[0][1]
    binWidth = importData[0][2]
    Rg = importData[0][3][:, newaxis]
    Rg_plot = linspace(min(Rg) - 20, max(Rg) + 20, 1000)[:, newaxis]
        
    grid = GridSearchCV(KernelDensity(),
                        {'bandwidth' : linspace(1, 5, 50)},
                        cv = 20)
    grid.fit(Rg)
    print('Best bandwidth: %r' % grid.best_params_)

    kde = KernelDensity(kernel = 'gaussian', bandwidth = grid.best_params_['bandwidth']).fit(Rg)
    log_dens = kde.score_samples(Rg_plot)
     
    fig, ax = plt.subplots(2,1, sharex = True, sharey = True)
    ax[0].hist(Rg, len(myBins - 1), normed = True)
    ax[1].fill(Rg_plot, exp(log_dens), fc='#AAAAFF')
    plt.show()"""

    # Test case 11: Test bumping path points
    """import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    numSegments, pLength = 200, 25
    myChain = WormlikeChain(numSegments, pLength)
    path = myChain.path
    bumpedPath = bumpPoints(path, 5)
    
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax.plot(path[:,0], path[:,1], path[:,2], linewidth = 2.0)
    plt.title('A')
    ax2 = fig.add_subplot(212, projection='3d')
    ax2.plot(bumpedPath[:,0], bumpedPath[:,1], bumpedPath[:,2], 'go', alpha = 0.5)
    plt.title('B')
    plt.show()"""

    # Test case 12: Test parallel collector
    from numpy import ones, append, array, concatenate
    C1, LP1 = meshgrid(arange(10, 60, 5), arange(10, 105, 5))
    C2, LP2 = meshgrid(arange(30, 65, 5), arange(105, 205, 5))
    C3, LP3 = meshgrid(arange(60, 100, 10), arange(10, 220, 20))
    C4, LP4 = meshgrid(array([20]), arange(110, 210, 20))

    C = concatenate((C1.flatten(), C2.flatten(), C3.flatten(), C4.flatten()))
    LP = concatenate((LP1.flatten(), LP2.flatten(), LP3.flatten(), LP4.flatten()))

    kwargs = {}
    kwargs['numPaths'] = 100000 # Number of paths per pair of walk parameters
    kwargs['pathLength'] =  24000 * (random(kwargs['numPaths']) - 0.5) + 27000 # bp in walk
    kwargs['linDensity'] = C  # bp / nm
    kwargs['persisLength'] = LP # nm 
    kwargs['segConvFactor'] = 2.5 # segments / min persisLen
    kwargs['nameDB'] = 'rw_' + dateStr
    kwargs['locPrecision'] = 2.12 # nm
    kwargs['fullSpecParam'] = True

    tic = time.time()
    myCollector = WLCCollector(**kwargs)
    toc = time.time()
    print('Total processing time: %f' % (toc - tic))

    """simResults = loadModel([kwargs['nameDB']])

    for key in simResults:
        Rg = simResults[key][0]
        c, lp = key
        RgTheory = WLCRg(c, lp, kwargs['pathLength'][0])

        print(dedent('''
                     c=%0.1f, lp=%0.1f
                     The mean of the simulated distribution is %f.
                     The mean theoretical gyration radius is %f.'''
                     % (c, lp, mean(Rg), RgTheory)))"""
        
