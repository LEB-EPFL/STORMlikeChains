"""Classes for simulating random walks in two or three dimensions.

"""

__author__ = 'Kyle M. Douglass'
__version__ = '0.1'
__email__ = 'kyle.m.douglass@gmail.com'

from math import modf
from numpy import pi, cos, sin, arccos
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
    
#    def _savePath():

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
        2D array of floats describing x,y points in the path.
        
        """
    def __init__(self,
                 numSegments,
                 pLength,
                 initPoint = array([1, 0, 0])):
        self.numSegments = numSegments
        self.pLength = pLength

        self.path = initPoint
        self._makePath()

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
        numSegFrac, numSegInt = modf(numSegments)
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

    def _makeNewPath(self, initPoint = array([1, 0, 0])):
        """Clears current path and makes a new one.

        Parameters
        ----------
        initPoint : array of float
            Coordinates of the first point.

        """
        self.path = initPoint
        self._makePath()

class Analyzer():
    """Analyzes paths for filtering and computing statistics.

    """

class Collector():
    """Counts the number of paths and checks for stop conditions.

    Parameters
    ----------
    numPaths : int
        The number of paths to collect before stopping the simulation.

    Attributes
    ----------
    myPath : path object
        The path for generating walks.
    """

    def __init__(self, numPaths):
        self.numPaths = numPaths

class WLCCollector(Collector):
    """Collector for the wormlike chain.

    """
    def __init__(self, numPaths, linDensity, pLength, segConv):
        super().__init__(numPaths)
        
        

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
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    numSegments, pLength = 1000.1, 25
    myChain = WormlikeChain(numSegments, pLength)
    path = myChain.path
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(path[:,0], path[:,1], path[:,2])
    plt.show()
    
