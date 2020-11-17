from numpy.random import uniform
from numpy import zeros, ndenumerate
import os
import sys

from min_bisection import input_schema


class GraphException(Exception):
    pass


def verify(b, msg=''):
    if not b:
        raise GraphException(msg)


class Graph:
    """Randomly generated graph with two equal sized clusters."""

    def __init__(self, n, p, q):
        """Generate our graph. The density of each cluster is controlled by p
        and the density of edges connecting the clusters is controlled by q.

        :param n: Number of vertices to generate in this graph
        :param p: The likelihood that a given node shares an edge with another given
        node within its cluster
        :param q: The likelihood that a given node shares an edge with a given
        node in another cluster.
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
        self.a = a + a.transpose()

    def save(self, fldr, cut_proportion=.1):
        """Create model ready data for min_bisection.py by saving the adjacency
        matrix to one csv and parameters for the model to another.

        :param fldr: what folder to save the csv's to
        :param cut_proportion: what proportion of initially infeasible constraints
        to add to the model at each solve in min_bisection.py
        :return:
        """
        verify(os.path.isdir(fldr), 'fldr should be an existing directory')
        verify(isinstance(cut_proportion, int) or isinstance(cut_proportion, float),
               'please ensure cut_proportion is an integer or a float')
        verify(0 <= cut_proportion <= 1,
               'please ensure cut_proportion is a value between 0 and 1 inclusively')

        ipt = input_schema.TicDat()
        for (i, j), v in ndenumerate(self.a):
            if i < j:
                ipt.a[(i, j)] = v
        ipt.parameters["Cut Proportion"] = cut_proportion

        input_schema.csv.write_directory(ipt, fldr, allow_overwrite=True)


if __name__ == '__main__':
    graph = Graph(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))
    graph.save(sys.argv[4])
