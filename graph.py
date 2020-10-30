from numpy.random import uniform
from numpy import zeros


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






