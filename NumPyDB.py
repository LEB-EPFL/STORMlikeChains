#!/usr/bin/env python
"""
Efficient database for NumPy objects.

Original version from SciTools, https://code.google.com/p/scitools/
Modified by Kyle M. Douglass to work with pickling and Python 3.
kyle.m.douglass@gmail.com
"""

import sys, os, pickle

class NumPyDB:
    def __init__(self, database_name, mode='store'):
        self.filename = database_name
        self.dn = self.filename + '.dat' # NumPy array data
        self.pn = self.filename + '.map' # positions & identifiers
        if mode == 'store':
            # bring files into existence:
            fd = open(self.dn, 'w');  fd.close()
            fm = open(self.pn, 'w');  fm.close()
        elif mode == 'load':
            # check if files are there:
            if not os.path.isfile(self.dn) or \
               not os.path.isfile(self.pn):
                raise IOError("Could not find the files %s and %s" %\
                              (self.dn, self.pn))
            # load mapfile into list of tuples:
            fm = open(self.pn, 'r')
            lines = fm.readlines()
            self.positions = []
            for line in lines:
                # first column contains file positions in the
                # file .dat for direct access, the rest of the
                # line is an identifier
                c = line.split()
                # append tuple (position, identifier):
                self.positions.append((int(c[0]),
                                       ' '.join(c[1:]).strip()))
            fm.close()

    def locate(self, identifier, bestapprox=None): # base class
        """
        Find position in files where data corresponding
        to identifier are stored.
        bestapprox is a user-defined function for computing
        the distance between two identifiers.
        """
        identifier = identifier.strip()
        # first search for an exact identifier match:
        selected_pos = -1
        selected_id = None
        for pos, id in self.positions:
            if id == identifier:
                selected_pos = pos;  selected_id = id; break
        if selected_pos == -1: # 'identifier' not found?
            if bestapprox is not None:
                # find the best approximation to 'identifier':
                min_dist = \
                    bestapprox(self.positions[0][1], identifier)
                for pos, id in self.positions:
                    d = bestapprox(id, identifier)
                    if d <= min_dist:
                        selected_pos = pos;  selected_id = id
                        min_dist = d
        return selected_pos, selected_id

    def dump(self, a, identifier):  # empty base class func.
        """Dump NumPy array a with identifier."""
        raise NameError("dump is not implemented; must be impl. in subclass")

    def load(self, identifier, bestapprox=None):
        """Load NumPy array with identifier or find best approx."""
        raise NameError("load is not implemented; must be impl. in subclass")

class NumPyDB_pickle (NumPyDB):
    """Use basic Pickle class."""

    def __init__(self, database_name, mode='store'):
        NumPyDB.__init__(self,database_name, mode)

    def dump(self, a, identifier):
        """Dump NumPy array a with identifier."""
        fd = open(self.dn, 'ab');  fm = open(self.pn, 'a')
        fm.write("%d\t\t %s\n" % (fd.tell(), identifier))
        #print('%r' % a)
        pickle.dump(a, fd)
        fd.close();  fm.close()

    def load(self, identifier, bestapprox=None):
        """
        Load NumPy array with a given identifier. In case the
        identifier is not found, bestapprox != None means that
        an approximation is sought. The bestapprox argument is
        then taken as a function that can be used for computing
        the distance between two identifiers id1 and id2.
        """
        pos, id = self.locate(identifier, bestapprox)
        if pos < 0: return None, "not found"
        fd = open(self.dn, 'rb')
        fd.seek(pos)
        a = pickle.load(fd)
        fd.close()
        return a, id

def float_dist(id1, id2):
    """
    Compute distance between two identities for NumPyDB.
    Assumption: id1 and id2 are real numbers (but always sent
    as strings).
    This function is typically used when time values are
    used as identifiers.
    """
    return abs(float(id1) - float(id2))


def _test_dist(id1, id2):
    """
    Return distance between identifiers id1 and id2.
    The identifiers are of the form 'time=some number'.
    """
    t1 = id1[5:];  t2 = id2[5:]
    d = abs(float(t1) - float(t2))
    return d

if __name__ == '__main__':
    print('hello!')

