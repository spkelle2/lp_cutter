import gurobipy as gu
import numpy as np
import random
import sys


def solve(n, p, q, cut_proportion):
    """

    :param n: size of our adjacency matrix (n x n)
    :param p: likelihood of edge within cluster
    :param q: likelihood of edge between clusters
    :param cut_proportion: what proportion of total constraints to select from
    those violated to add to our model
    :return:
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

    # create index sets for triangle inequality constraints
    tri1 = [(i, j, k) for i in indices for j in indices[i + 1:] for k in indices
            if i != k != j]
    tri2 = [(i, j, k) for i in indices for j in indices[i + 1:] for k in
            indices[j + 1:]]
    for index_set in [tri1, tri2]:
        random.shuffle(index_set)
    cut_size = int(cut_proportion*(len(tri1) + len(tri2)))

    # build our model
    mdl = gu.Model("min bisection")

    # variables
    x = {(i, j): mdl.addVar(vtype='B', name=f'{i}_{j}') for i in indices
         for j in indices if i != j}

    # objective
    mdl.setObjective(gu.quicksum(a[i, j] * x[i, j] for i in indices
                                 for j in indices if i < j),
                     sense=gu.GRB.MINIMIZE)

    # (3) Equal partition constraint
    mdl.addConstr(gu.quicksum(x[i, j] for (i, j) in x if i < j) == n**2/4,
                  name='Equal Partitions')

    # (1) Add randomly 50 of the 1st triangle inequality constraints
    for (i, j, k) in tri1[:50]:
        mdl.addConstr(x[i, j] <= x[i, k] + x[j, k], name=f'{i}_{j}_{k}_tri1')
    tri1 = tri1[50:]

    # (2) Add randomly 50 of the 2st triangle inequality constraints
    for (i, j, k) in tri2[:50]:
        mdl.addConstr(x[i, j] + x[i, k] + x[j, k] <= 2, name=f'{i}_{j}_{k}_tri2')
    tri2 = tri2[50:]

    mdl.optimize()
    assert mdl.status == gu.GRB.OPTIMAL, 'small initial solve should make solution'

    while True:
        # find all constraints that are violated. note any violated constraint
        # has same depth, so don't sweat finding the "most" violated ones
        inf1 = [(i, j, k) for (i, j, k) in tri1 if x[i, j] > x[i, k] + x[j, k]]
        inf2 = [(i, j, k) for (i, j, k) in tri2 if x[i, j] + x[i, k] + x[j, k] > 2]
        if not inf1 or inf2:
            break

        # im sure there's a better way to select from our most violated cuts
        # but this is quick and easy for now
        for (i, j, k) in inf1[:cut_size//2]:
            mdl.addConstr(mdl.addConstr(x[i, j] <= x[i, k] + x[j, k],
                                        name=f'{i}_{j}_{k}_tri1'))
            tri1.remove((i, j, k))

        for (i, j, k) in inf2[:cut_size//2]:
            mdl.addConstr(mdl.addConstr(x[i, j] + x[i, k] + x[j, k] <= 2,
                                        name=f'{i}_{j}_{k}_tri2'))
            tri2.remove((i, j, k))

        mdl.optimize()
        assert mdl.status == gu.GRB.OPTIMAL, f"model ended up as: {mdl.status}"

    print()


if __name__ == '__main__':
    solve(n=int(sys.argv[1]), p=float(sys.argv[2]), q=float(sys.argv[3]),
          cut_proportion=float(sys.argv[4]))