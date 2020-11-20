import gurobipy as gu
import numpy as np
import random
import sys
import time


def create_adjacency_matrix(n, p, q):
    """Generate our graph. The density of each cluster is controlled by p
    and the density of edges connecting the clusters is controlled by q.

    :param n: Number of vertices to generate in this graph
    :param p: The likelihood that a given node shares an edge with another given
    node within its cluster
    :param q: The likelihood that a given node shares an edge with a given
    node in another cluster.
    :return a: 2-D array where a[i,j]=1 if edge exists between i and j, else 0
    """
    indices = range(n)

    # create our adjacency matrix
    cluster1 = indices[:n // 2]
    cluster2 = indices[n // 2:]
    a = np.zeros((n, n))

    # complete adjacency matrix for upper right entries excluding diagonal
    for i in indices:
        for j in indices[i + 1:]:
            if {i, j} <= set(cluster1) or {i, j} <= set(cluster2):
                a[i, j] = int(p > np.random.uniform())  # edge in same cluster with prob p
            else:
                a[i, j] = int(q > np.random.uniform())
    # then copy them to bottom left
    a = a + a.transpose()
    return a


def create_constraint_indices(indices):
    """create dictionary of triangle inequality constraints to be added to the model
    {(index_triple, constraint_number): amount_violated}

    :param indices: a range(n) representing the numbers to iterate over
    :return c: a dictionary keyed by index_triple and constraint_number with value
    later to be set as how violated the index triple's constraint is for the given
    constraint number
    """

    tri1 = {((i, j, k), 1): 0 for i in indices for j in indices[i + 1:] for k in
            indices if i != k != j}
    tri2 = {((i, j, k), 2): 0 for i in indices for j in indices[i + 1:] for k in
            indices[j + 1:]}
    c = {**tri1, **tri2}
    return c


class MinBisect:

    def __init__(self, n, p, q, cut_proportion):
        """Create our adjacency matrix and constraint indexes and declare all
        other needed attributes

        :param n: size of our adjacency matrix (n x n)
        :param p: likelihood of edge within cluster
        :param q: likelihood of edge between clusters
        :param cut_proportion: what proportion of total constraints to select from
        those violated to add to our model
        :return:
        """
        self.n = n
        self.indices = range(n)
        self.a = create_adjacency_matrix(n, p, q)
        self.c = create_constraint_indices(self.indices)
        self.cut_size = int(cut_proportion*len(self.c))
        self.mdl = None
        self.x = None

    def instantiate_model(self):
        """Does everything that solving iteratively and at once will share, namely
        instantiating the model and variables as well as setting the objective
        and equal partition constraint.

        :return:
        """
        self.mdl = gu.Model("min bisection")  # check to make sure this gives empty model
        self.mdl.setParam(gu.GRB.Param.Method, 1)

        # variables
        self.x = {(i, j): self.mdl.addVar(ub=1, name=f'{i}_{j}') for i in self.indices
                  for j in self.indices if i != j}

        # objective
        self.mdl.setObjective(gu.quicksum(self.a[i, j] * self.x[i, j] for (i, j)
                                          in self.x if i < j), sense=gu.GRB.MINIMIZE)

        # (3) Equal partition constraint
        self.mdl.addConstr(gu.quicksum(self.x[i, j] for (i, j) in self.x if i < j)
                           == self.n ** 2 / 4, name='Equal Partitions')

    def add_triangle_inequality(self, i, j, k, t):
        """Adds a triangle inequality to the model and removes its index from
        future candidate constraints.

        :param i: ith index
        :param j: jth index
        :param k: kth index
        :param t: whether this is constraint type 1 or 2
        :return:
        """
        if t == 1:
            # (1) 1st triangle inequality constraint
            self.mdl.addConstr(self.x[i, j] <= self.x[i, k] + self.x[j, k],
                               name=f'{i}_{j}_{k}_tri1')
        else:  # t == 2:
            # (2) 2nd triangle inequality constraint
            self.mdl.addConstr(self.x[i, j] + self.x[i, k] + self.x[j, k] <= 2,
                               name=f'{i}_{j}_{k}_tri2')
        del self.c[(i, j, k), t]

    def solve_once(self):
        """Solves the model with all constraints added at once

        :return:
        """
        self.instantiate_model()
        for ((i, j, k), t) in self.c.keys():  # may need to make a list first
            self.add_triangle_inequality(i, j, k, t)

        self.mdl.optimize()
        assert self.mdl.status == gu.GRB.OPTIMAL, 'small initial solve should make solution'

    def solve_iteratively(self):
        """Solve the model by feeding in only the top most violated constraints,
        and repeat until no violated constraints remain

        :return:
        """

        self.instantiate_model()
        # Add randomly 100 of the triangle inequality constraints
        for ((i, j, k), t) in random.sample(self.c.keys(), min(100, len(self.c))):
            self.add_triangle_inequality(i, j, k, t)

        self.mdl.optimize()
        assert self.mdl.status == gu.GRB.OPTIMAL, 'small initial solve should make solution'

        while True:
            # find how much each constraint is violated
            # no need to normalize since same size vectors
            self.c = {((i, j, k), t): self.x[i, j].x - self.x[i, k].x - self.x[j, k].x
                      if t == 1 else self.x[i, j].x + self.x[i, k].x + self.x[j, k].x - 2
                      for ((i, j, k), t) in self.c}
            inf = [k for k in sorted(self.c, key=self.c.get, reverse=True) if
                   self.c[k] > 0][:self.cut_size]
            if not inf:
                break

            for ((i, j, k), t) in inf:
                self.add_triangle_inequality(i, j, k, t)

            self.mdl.optimize()
            assert self.mdl.status == gu.GRB.OPTIMAL, f"model ended up as: {self.mdl.status}"


if __name__ == '__main__':
    start = time.time()
    mb = MinBisect(n=int(sys.argv[1]), p=float(sys.argv[2]), q=float(sys.argv[3]),
                   cut_proportion=float(sys.argv[4]))
    mb.solve_iteratively()
    print(f'solve time: {time.time() - start} seconds')