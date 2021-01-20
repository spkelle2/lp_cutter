import gurobipy as gu
from math import floor
import random

from min_bisection import create_constraint_indices, create_adjacency_matrix, \
    MinBisect, profilable_random
from profiler import profile_run_time


# @profile_run_time(sort_by='tottime', lines_to_print=20, strip_dirs=True)
def solve_iterative_min_bisect(n, p, q, cut_size, threshold_proportion=None, a=None):
    """for a given n, p, q, cut_size, and optional threshold_proportion, run an
    iterative solve. Optionally accepts the adjacency matrix used in another run
    for comparison. If given a threshold_proportion, will only search through
    as many unadded constraints in each iteration as is required to find
    <cut_size> more violated than <threshold_proportion> proportion of violated
    constraints. If no <threshold_proportion> is given, each iteration will
    seek out the <cut_size> most violated constraints. Returns the optimal
    solution, optimal objective value, and the adjacency matrix that was used"""

    # code that would otherwise get repeated
    def _add_triangle_inequality(i, j, k, t):
        """Adds a triangle inequality to the model and removes its index from
        future candidate constraints."""

        assert t in [1, 2, 3, 4], 'constraint type should be 1, 2, 3, or 4'
        if t == 1:
            # (1) 1st triangle inequality constraint, type 1
            mdl.addConstr(x[j, k] <= x[i, j] + x[i, k],
                          name=f'{i}_{j}_{k}_tri1')
        elif t == 2:
            # (1) 1st triangle inequality constraint, type 2
            mdl.addConstr(x[i, k] <= x[i, j] + x[j, k],
                          name=f'{i}_{j}_{k}_tri2')
        elif t == 3:
            # (1) 1st triangle inequality constraint, type 3
            mdl.addConstr(x[i, j] <= x[i, k] + x[j, k],
                          name=f'{i}_{j}_{k}_tri3')
        else:  # t == 4:
            # (2) 2nd triangle inequality constraint
            mdl.addConstr(x[i, j] + x[i, k] + x[j, k] <= 2,
                          name=f'{i}_{j}_{k}_tri4')
        c.remove(((i, j, k), t))
        return

    def _get_cut_depth(i, j, k, t):
        """find how much each constraint is violated. don't worry about
        normalizing since each vector has the same norm"""

        if t == 1:
            try:
                return v[j, k] - v[i, j] - v[i, k]
            except KeyError:
                _get_vals(i, j, k)
                return v[j, k] - v[i, j] - v[i, k]
        elif t == 2:
            try:
                return v[i, k] - v[i, j] - v[j, k]
            except KeyError:
                _get_vals(i, j, k)
                return v[i, k] - v[i, j] - v[j, k]
        elif t == 3:
            try:
                return v[i, j] - v[i, k] - v[j, k]
            except KeyError:
                _get_vals(i, j, k)
                return v[i, j] - v[i, k] - v[j, k]
        else:  # t == 4:
            try:
                return v[i, j] + v[i, k] + v[j, k] - 2
            except KeyError:
                _get_vals(i, j, k)
                return v[i, j] + v[i, k] + v[j, k] - 2

    def _get_vals(i, j, k):
        for (a, b) in [(j, k), (i, j), (i, k)]:
            if (a, b) not in v:
                v[a, b] = x[a, b].x
        return

    # stuff to use later
    tolerance = .0001
    first_iteration_cuts = 100
    indices = range(n)
    a = a if a is not None else create_adjacency_matrix(n, p, q)
    c = create_constraint_indices(indices)
    current_threshold = None
        
    # model
    mdl = gu.Model("min bisection")
    mdl.setParam(gu.GRB.Param.LogToConsole, 0)
    x = {(i, j): mdl.addVar(ub=1, name=f'x_{i}_{j}') for i in indices for j in
         indices if i < j}
    mdl.setObjective(gu.quicksum(a[i, j] * x[i, j] for (i, j) in x),
                     sense=gu.GRB.MINIMIZE)
    mdl.addConstr(gu.quicksum(x[i, j] for (i, j) in x) == n ** 2 / 4,
                  name='equal_partitions')

    # solve first 100 constraints
    inf = random.sample(c, min(first_iteration_cuts, len(c)))
    for ((i, j, k), t) in inf:
        _add_triangle_inequality(i, j, k, t)
    mdl.optimize()

    # solve iterations
    while True:
        d = {}
        v = {}
        if current_threshold:

            # _recalibrate_cut_depths_by_threshold_proportion
            count = 0
            for ((i, j, k), t) in c:
                cut_depth = _get_cut_depth(i, j, k, t)
                if cut_depth >= current_threshold:
                    d[(i, j, k), t] = cut_depth
                    count += 1
                    if count == cut_size:
                        break

            inf = list(d.keys())
            if len(inf) < cut_size:
                current_threshold = None
        else:

            # _recalibrate_cut_depths original (since no random search proportion)
            for ((i, j, k), t) in c:
                cut_depth = _get_cut_depth(i, j, k, t)
                if cut_depth > tolerance:
                    d[(i, j, k), t] = cut_depth

            inf = sorted(d, key=d.get, reverse=True)[:cut_size]  # descending

            # only find new threshold if we were given a proportion to begin with and
            # there are at least <cut_size> constraints currently with the new depth or more
            if threshold_proportion and (cut_size < len(d) * (1 - threshold_proportion)):
                key = sorted(d, key=d.get)[floor(len(d) * threshold_proportion)]  # ascending
                current_threshold = d[key]

            if not inf:  # end if no violated constraints found
                x = {(i, j): v.x for (i, j), v in x.items()}
                return x, mdl.ObjVal, a

        for ((i, j, k), t) in inf:
            _add_triangle_inequality(i, j, k, t)
        mdl.optimize()


# @profile_run_time(sort_by='tottime', lines_to_print=10, strip_dirs=True)
def profilable_slim(mbs):
    for i, mb in enumerate(mbs):
        print(f'test {i + 1 + len(mbs)}')
        solve_iterative_min_bisect(n=60, p=.5, q=.2, cut_size=1000, a=mb.a,
                                   threshold_proportion=.9)


if __name__ == '__main__':
    mbs = profilable_random(5)
    profilable_slim(mbs)

