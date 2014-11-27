"""Classes for simulating random walks in three dimensions.

"""

__author__ = 'Kyle M. Douglass'
__version__ = '0.1'
__email__ = 'kyle.m.douglass@gmail.com'

from math import modf
from textwrap import dedent
from numpy import pi, cos, sin, arccos, meshgrid, sum, var, zeros
from numpy import array, cross, concatenate, hstack, vstack, cumsum
from numpy.random import randn, random
from numpy.linalg import norm

class Path():
    def _normVector(self, vectors):
        """Normalize an array of vectors.

        Parameters
        ----------
        vectors : array of double
            3 x N array of doubles with x, y, and z coordinates.

        Returns
        -------
        vectorsNormed : array of double
            The normalized vectors from the original vectors array.
        """
        if vectors.shape[0] != 3:
            raise SizeException(
                'The number of rows in vectors is not 3.')
        
        normFactors = norm(vectors, axis = 0)
        vectorsNormed = vectors / normFactors

        return vectorsNormed
                
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
        angDisp = self.pLength ** (-0.5) * randn(self.numSegments - 1)
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

        currPoint = initPoint
        for ctr in range(len(tanPlaneDisp)):
            # Create a displacement in the plane tangent to currPoint
            dispVector = cross(currPoint, randVecs[:,ctr])

            # Check if displacement and currPoint vectors are parallel
            while norm(dispVector) == 0:
                newRandVec = self._randPointSphere(1)
                dispVector = cross(currPoint, newRandVec)

            # Move the currPoint vector in the tangent plane
            dispVector = self._normVector(dispVector) \
                    * tanPlaneDisp[ctr]
            
            # Back project new point onto sphere
            projDistance = 1 - cos(angDisp[ctr])
            nextPoint = (1 - projDistance) * currPoint + dispVector
                        
            # Append nextPoint to array of points on the path
            self.path = vstack((self.path, nextPoint))
            currPoint = nextPoint

        # Add up the vectors in path to create the polymer
        self.path = cumsum(self.path, axis = 0)

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
    """Analyzes paths for filtering and computing statistics.

    """
    def __init__(self):
        print('Hello. I am an analyzer.')

    def computeRg(self, myPath):
        """Compute the radius of gyration of a path.

        computeRg() calculates the radius of gyration of a Path object
        and assigns it to a field within the same Path object.

        Parameters
        ----------
        myPath : Path or child of Path
            Path for which the radius of gyration is to be determined
        """
        pathShape = myPath.path.shape
        if pathShape[1] != 2 and pathShape[1] != 3:
            errorStr = dedent('''
            Error: Path array has %d columns.
            A path must have either 2 or 3 columns.
            For 2D walks, the columns are the x and y coordinates.
            For 3D walks, the columns are the x, y, and z coordinates.
            ''' % pathShape[1])
            
            raise SizeException(errorStr)

        secondMoments = var(myPath.path, axis = 0)
        Rg = (sum(secondMoments)) ** (0.5)

        return Rg
        
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

    def __init__(self, numPaths, pathLength, segConvFactor = 1):
        if numPaths != len(pathLength):
            errorStr = dedent('''
                Number of paths input: %r
                Length of path lengths vector: %r

                These values must be equal and integers.'''
                % (numPaths, len(pathLength)))
            
            raise SizeException(errorStr)
        
        self.numPaths = numPaths
        self._segConvFactor = segConvFactor
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
                 myAnalyzer = Analyzer()):
        super().__init__(numPaths, pathLength, segConvFactor)
        self._linDensity = self._convSegments(linDensity, False)
        self._persisLength = self._convSegments(persisLength, True)
        self.__pathLength = pathLength
        self._myAnalyzer = myAnalyzer

        self._startCollector()

    def _startCollector(self):
        """Begin collecting wormlike chain conformation statistics.

        """
        linDensity, persisLength = meshgrid(self._linDensity,
                                            self._persisLength)
        # Loop over all combinations of density and persistence length
        for c, lp in zip(linDensity.flatten(),
                         persisLength.flatten()):
            #print('Density: %r, Persistence length: %r'
            #      %(c, lp))

            numSegments = self.__pathLength / c
            #print('Number of segments: %r' % numSegments)
            
            # Does the collector already have a path object?
            if not hasattr(self, '_myPath'):
                self._myPath = WormlikeChain(numSegments[0], lp)

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

            print('%r' % Rg)

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

    myCollector = WLCCollector(numPaths, pathLength, linDensity, persisLength, segConvFactor)"""

    # Test case 5: Create an analyzer and compute the Rg of a WLC. 
    numPaths = 5 # Number of paths per pair of walk parameters
    pathLength = (2000) **(0.5) * randn(numPaths) + 25000 # bp in walk
    linDensity = array([40, 50]) # bp / nm
    persisLength = array([15, 20, 25]) # nm
    segConvFactor = 10 / min(persisLength) # segments / min persisLen

    myCollector = WLCCollector(numPaths, pathLength, linDensity, persisLength, segConvFactor)
