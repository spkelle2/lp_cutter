from numpy.random import uniform
from numpy import zeros, ndarray, unique
import os
from pandas import DataFrame
import sys

from min_bisection import input_schema


class GraphException(Exception):
    pass


def verify(b, msg=''):
    if not b:
        raise GraphException(msg)


def two_clustered_graph(n, p, q):
    """Randomly generate a graph with two equal sized clusters. The density of
    each cluster is controlled by p and the density of edges connecting the
    clusters is controlled by q.

    :param n: Number of vertices to generate in this graph
    :param p: The likelihood that a given node shares an edge with another given
    node within its cluster
    :param q: The likelihood that a given node shares an edge with a given
    node in another cluster.
    :return A: Adjacency dictionary A[i][j] where i indexes rows, j indexes
    columns, and the return value is 1 should node i share an edge with node j
    and 0 if not.
    """
    verify(isinstance(n, int) and n > 0,
           'please make sure n is an integer greater than 0')
    for var, val in {'p': p, 'q': q}.items():
        verify(isinstance(val, int) or isinstance(val, float),
               f'please ensure {var} is an integer or a float')
        verify(0 <= val <= 1,
               f'please ensure {var} is a value between 0 and 1 inclusively')

        indices = range(n)
        cluster1 = indices[:n//2]
        cluster2 = indices[n//2:]
        a = zeros((n, n))

        # complete adjacency matrix for upper right entries excluding diagonal
        for i in indices:
            for j in indices[i+1:]:
                if {i, j} <= set(cluster1) or {i, j} <= set(cluster2):
                    a[i, j] = int(p > uniform())  # edge in same cluster with prob p
                else:
                    a[i, j] = int(q > uniform())
        # then copy them to bottom left
        a = a + a.transpose()

        return a


def save_graph(a, fldr):
    """ Take an adjacency matrix and save it to a csv

    :param a: an adjacency matrix
    :param fldr: where to save the matrix
    :return:
    """
    verify(isinstance(a, ndarray), 'the matrix should be a numpy ndarray')
    verify(a.shape[0] == a.shape[1], 'the matrix should be square')
    verify(set(unique(a)) == {0, 1}, 'values can only be 0 and 1')
    verify(a.trace() == 0, 'no values should exist on diagonal')
    verify((a == a.T).all(), 'transpose should be equal')
    verify(os.path.isdir(fldr), 'fldr should be an existing directory')

    ipt = input_schema.TicDat()
    ipt.a = DataFrame(a, columns=[str(i) for i in range(a.shape[0])], dtype=int)
    ipt.parameters["Cut Proportion"] = .1

    input_schema.csv.write_directory(ipt, fldr, allow_overwrite=True)


if __name__ == '__main__':
    a = two_clustered_graph(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))
    save_graph(a, sys.argv[4])
