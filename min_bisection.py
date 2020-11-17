import gurobipy as gu
import numpy as np
import random
import sys
import time


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

    # create dictionary of triangle inequality constraints not yet added to model
    # {(index_triple, constraint_number): amount_violated}
    tri1 = {((i, j, k), 1): 0 for i in indices for j in indices[i + 1:] for k in
            indices if i != k != j}
    tri2 = {((i, j, k), 2): 0 for i in indices for j in indices[i + 1:] for k in
            indices[j + 1:]}
    c = {**tri1, **tri2}
    cut_size = int(cut_proportion*len(c))

    # build our model
    mdl = gu.Model("min bisection")
    mdl.setParam(gu.GRB.Param.Method, 1)

    # variables
    x = {(i, j): mdl.addVar(ub=1, name=f'{i}_{j}') for i in indices
         for j in indices if i != j}

    # objective
    mdl.setObjective(gu.quicksum(a[i, j] * x[i, j] for (i, j) in x if i < j),
                     sense=gu.GRB.MINIMIZE)

    # (3) Equal partition constraint
    mdl.addConstr(gu.quicksum(x[i, j] for (i, j) in x if i < j) == n**2/4,
                  name='Equal Partitions')

    def add_triangle_inequality(i, j, k, t):
        if t == 1:
            # (1) 1st triangle inequality constraint
            mdl.addConstr(x[i, j] <= x[i, k] + x[j, k], name=f'{i}_{j}_{k}_tri1')
        else:  # t == 2:
            # (2) 2nd triangle inequality constraint
            mdl.addConstr(x[i, j] + x[i, k] + x[j, k] <= 2, name=f'{i}_{j}_{k}_tri2')
        del c[(i, j, k), t]

    # Add randomly 100 of the triangle inequality constraints
    for ((i, j, k), t) in random.sample(c.keys(), min(100, len(c))):
        add_triangle_inequality(i, j, k, t)

    mdl.optimize()
    assert mdl.status == gu.GRB.OPTIMAL, 'small initial solve should make solution'

    while True:
        # find all constraints that are violated. Could tolerance here...
        # no need to normalize since same size vectors
        c = {((i, j, k), t): x[i, j].x - x[i, k].x - x[j, k].x if t == 1 else
             x[i, j].x + x[i, k].x + x[j, k].x - 2 for ((i, j, k), t) in c}
        inf = [k for k in sorted(c, key=c.get, reverse=True) if c[k] > 0][:cut_size]
        if not inf:
            break

        for ((i, j, k), t) in inf:
            add_triangle_inequality(i, j, k, t)

        mdl.optimize()
        assert mdl.status == gu.GRB.OPTIMAL, f"model ended up as: {mdl.status}"

    print()


if __name__ == '__main__':
    start = time.time()
    solve(n=int(sys.argv[1]), p=float(sys.argv[2]), q=float(sys.argv[3]),
          cut_proportion=float(sys.argv[4]))
    print(f'solve time: {time.time() - start} seconds')