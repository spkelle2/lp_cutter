import gurobipy as gu
from math import ceil, floor
import numpy as np
import random
from ticdat import TicDatFactory
import time

solution_schema = TicDatFactory(
    run_stats=[['solve_id', 'solve_type', 'method', 'warm_start', 'sub_solve_id'],
               ['n', 'p', 'q', 'cut_type', 'cut_value', 'cuts_sought',
                'cuts_added', 'min_order', 'min_proportion', 'proportion_used',
                'variables', 'constraints', 'cpu_time']],
    summary_stats=[['solve_id', 'solve_type', 'method', 'warm_start'],
                   ['n', 'p', 'q', 'cut_type', 'cut_value', 'min_order',
                    'min_proportion', 'max_variables', 'max_constraints',
                    'total_cpu_time', 'gurobi_cpu_time', 'non_gurobi_cpu_time',
                    'objective_value']]
)


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

    c = {((i, j, k), idx) for i in indices for j in indices[i + 1:]
         for k in indices[j + 1:] for idx in [1, 2, 3, 4]}
    return c


class MinBisect:

    def __init__(self, n, p, q, cut_proportion=None, number_of_cuts=None,
                 solve_id=0, tolerance=.0001, log_to_console=0, log_file_base='',
                 write_mps=False, first_iteration_cuts=100, min_order=0):
        """Create our adjacency matrix and constraint indexes and declare all
        other needed attributes

        :param n: size of our adjacency matrix (n x n)
        :param p: likelihood of edge within cluster
        :param q: likelihood of edge between clusters
        :param cut_proportion: what proportion of total constraints to select from
        those violated to add to our model
        :param number_of_cuts: how many constraints to select from those violated
        to add to our model. Please only use at most one of number_of_cuts and
        cut_proportion
        :param solve_id: id to mark this run by in output data
        :param tolerance: how close to actually being satisfied a constraint
        must be for us to just go ahead and consider it satisfied
        :param log_to_console: 0 to run gurobi without printing outputs to console,
        1 otherwise
        :param log_file_base: base string to be shared between both log files.
        Blank creates no log file
        :param write_mps: whether or not to write out the .mps file for this experiment
        :param first_iteration_cuts: number of cuts to add to the first solve
        when solving iteratively
        :param min_order: how many orders of magnitude larger must the smallest
        set of constraints to search for violations be relative to how many cuts
        we intend to add at once.
        :return:
        """
        assert n % 2 == 0, 'n needs to be even'
        assert 0 <= p <= 1, 'p is probability'
        assert 0 <= q <= 1, 'q is probability'
        assert cut_proportion is None or (0 <= cut_proportion <= 1), 'cut proportion is ratio'
        assert number_of_cuts is None or number_of_cuts >= 1, 'need to have at least one cut'
        assert (cut_proportion or number_of_cuts), 'at least one'
        assert not (cut_proportion and number_of_cuts), 'not both'
        assert 0 <= tolerance < 1, 'tolerance should be between 0 and 1'
        assert log_to_console in [0, 1], 'gurobi requires log to console flag either 0 or 1'
        assert isinstance(write_mps, bool), 'write_mps must be boolean'
        assert isinstance(first_iteration_cuts, int) and first_iteration_cuts >= 0
        assert isinstance(min_order, int) and min_order >= 0

        self.n = n
        self.p = p
        self.q = q
        self.solve_id = solve_id
        self.tolerance = tolerance
        self.log_to_console = log_to_console
        self.log_file_base = log_file_base
        self.write_mps = write_mps

        self.indices = range(n)
        self.a = create_adjacency_matrix(n, p, q)
        self.c = None  # available cuts
        self.d = {}  # cut depths
        self.proportion = 1  # proportion of cuts to search
        self.cut_type = 'proportion' if cut_proportion else 'fixed'
        self.cut_value = cut_proportion if cut_proportion else number_of_cuts
        self.cut_size = None
        self.mdl = None
        self.x = None
        self.sub_solve_id = None
        self.variables = 0
        self.constraints = 0
        self.data = solution_schema.TicDat()
        self.solve_type = None
        self.inf = []
        self.first_iteration_cuts = first_iteration_cuts
        self.method = None
        self.warm_start = None
        self.search_proportions = None
        self.min_order = min_order

    @property
    def file_combo(self):
        return f'{self.solve_type}_{self.method}' \
               f'{"" if self.solve_type != "iterative" else f"_{self.warm_str}"}'

    @property
    def warm_str(self):
        return 'warm' if self.warm_start else 'cold'

    def _instantiate_model(self, method='dual'):
        """Does everything that solving iteratively and at once will share, e.g.
        instantiating the model and variables as well as setting the objective
        and equal partition constraint.

        :param method: 'dual' to solve each iteration with dual simplex,
        'auto' to let gurobi decide which solve method is best

        :return:
        """
        assert method in ['dual', 'auto'], 'solve with dual or let gurobi decide'

        self.method = method
        self.c = create_constraint_indices(self.indices)
        self.sub_solve_id = -1
        self.cut_size = len(self.c) if self.solve_type == 'once' else \
            floor(self.cut_value * len(self.c)) if self.cut_type == 'proportion' \
            else self.cut_value
        self.search_proportions = [
            10**x for x in range(ceil(np.log10(self.cut_size/len(self.c))), 0)
        ][self.min_order:] + [1]

        # model
        self.mdl = gu.Model("min bisection")
        self.mdl.setParam(gu.GRB.Param.LogToConsole, self.log_to_console)
        self.mdl.setParam(gu.GRB.Param.Method, 1 if self.method == 'dual' else -1)
        if self.log_file_base:
            self.mdl.setParam(gu.GRB.Param.LogFile,
                              f'{self.log_file_base}_{self.file_combo}.txt')

        # variables
        self.x = {(i, j): self.mdl.addVar(ub=1, name=f'x_{i}_{j}') for i in
                  self.indices for j in self.indices if i < j}

        # objective
        self.mdl.setObjective(gu.quicksum(self.a[i, j] * self.x[i, j] for (i, j)
                                          in self.x), sense=gu.GRB.MINIMIZE)

        # (3) Equal partition constraint
        self.mdl.addConstr(gu.quicksum(self.x[i, j] for (i, j) in self.x)
                           == self.n ** 2 / 4, name='equal_partitions')

    def _add_triangle_inequality(self, i, j, k, t):
        """Adds a triangle inequality to the model and removes its index from
        future candidate constraints.

        :param i: ith index
        :param j: jth index
        :param k: kth index
        :param t: whether this is constraint type 1 or 2
        :return:
        """
        assert t in [1, 2, 3, 4], 'constraint type should be 1, 2, 3, or 4'
        if t == 1:
            # (1) 1st triangle inequality constraint, type 1
            self.mdl.addConstr(self.x[j, k] <= self.x[i, j] + self.x[i, k],
                               name=f'{i}_{j}_{k}_tri1')
        elif t == 2:
            # (1) 1st triangle inequality constraint, type 2
            self.mdl.addConstr(self.x[i, k] <= self.x[i, j] + self.x[j, k],
                               name=f'{i}_{j}_{k}_tri2')
        elif t == 3:
            # (1) 1st triangle inequality constraint, type 3
            self.mdl.addConstr(self.x[i, j] <= self.x[i, k] + self.x[j, k],
                               name=f'{i}_{j}_{k}_tri3')
        else:  # t == 4:
            # (2) 2nd triangle inequality constraint
            self.mdl.addConstr(self.x[i, j] + self.x[i, k] + self.x[j, k] <= 2,
                               name=f'{i}_{j}_{k}_tri4')
        self.c.remove(((i, j, k), t))

    def _summary_profile(func):
        def wrapper(self, *args, **kwargs):
            solve_start = time.process_time()
            retval = func(self, *args, **kwargs)
            total_cpu_time = time.process_time() - solve_start
            # sum over sub solve indices for this combination of solve
            # solve_id, solve_type, method, warm_start, sub_solve_id
            gurobi_cpu_time = sum(d['cpu_time'] for (si, st, m, ws, ssi), d
                                  in self.data.run_stats.items() if
                                  self.solve_type == st and self.method == m and
                                  self.warm_str == ws)
            self.data.summary_stats[self.solve_id, self.solve_type, self.method,
                                    self.warm_str] = {
                'n': self.n,
                'p': self.p,
                'q': self.q,
                'cut_type': self.cut_type,
                'cut_value': self.cut_value,
                'min_order': self.min_order,
                'min_proportion': min(self.search_proportions),
                'max_constraints': self.mdl.NumConstrs,
                'max_variables': self.mdl.NumVars,
                'total_cpu_time': total_cpu_time,
                'gurobi_cpu_time': gurobi_cpu_time,
                'non_gurobi_cpu_time': total_cpu_time - gurobi_cpu_time,
                'objective_value': self.mdl.ObjVal
            }
            return retval

        return wrapper

    def _optimize(self):
        self.sub_solve_id += 1
        if self.write_mps:
            self.mdl.write(f'model_{self.file_combo}_{self.sub_solve_id}.mps')
        sub_solve_start = time.process_time()
        self.mdl.optimize()
        sub_solve_cpu_time = time.process_time() - sub_solve_start
        self.constraints = self.mdl.NumConstrs
        self.variables = self.mdl.NumVars
        self.data.run_stats[self.solve_id, self.solve_type, self.method,
                            self.warm_str, self.sub_solve_id] = {
            'n': self.n,
            'p': self.p,
            'q': self.q,
            'cut_type': self.cut_type,
            'cut_value': self.cut_value,
            'cuts_sought': len(self.inf) if self.solve_type == 'iterative' and
                           self.sub_solve_id == 0 else self.cut_size,
            'cuts_added': len(self.inf) if self.solve_type == 'iterative' else self.cut_size,
            'min_order': self.min_order,
            'min_proportion': min(self.search_proportions),
            'proportion_used': self.proportion,
            'constraints': self.mdl.NumConstrs,
            'variables': self.mdl.NumVars,
            'cpu_time': sub_solve_cpu_time
        }

    @_summary_profile
    def solve_once(self, method='auto'):
        """Solves the model with all constraints added at once

        :param method: 'dual' to solve the model with dual simplex,
        'auto' to let gurobi decide which solve method is best

        :return:
        """
        self.solve_type = 'once'
        self.warm_start = False
        self.proportion = 1
        self._instantiate_model(method)

        # make new object so loop not thrown off by deletion
        for ((i, j, k), t) in list(self.c):
            self._add_triangle_inequality(i, j, k, t)

        self._optimize()
        assert self.mdl.status == gu.GRB.OPTIMAL, 'once solve should have solution'

    def _recalibrate_cut_depths(self):
        """find how much each constraint is violated. don't worry about
        normalizing since each vector has the same norm

        for a constraint to be satisfied, its key should have a corresponding
        value <= 0. Changing this such that the value <= tolerance allows constraints
        close to being satisfied to still be considered satisfied

        :return:
        """
        for ((i, j, k), t) in random.sample(self.c, int(self.proportion*len(self.c)))\
                if self.proportion < 1 else self.c:
            if t == 1:
                cut_depth = self.x[j, k].x - self.x[i, j].x - self.x[i, k].x
            elif t == 2:
                cut_depth = self.x[i, k].x - self.x[i, j].x - self.x[j, k].x
            elif t == 3:
                cut_depth = self.x[i, j].x - self.x[i, k].x - self.x[j, k].x
            else:  # t == 4:
                cut_depth = self.x[i, j].x + self.x[i, k].x + self.x[j, k].x - 2
            if cut_depth > self.tolerance:
                self.d[(i, j, k), t] = cut_depth

    def _find_most_violated_constraints(self):
        """Find all constraint indices that represent infeasible constraints (i.e.
        have values greater than the tolerance) then take the <self.cut_size>
        most violated

        :return:
        """
        self.inf = sorted(self.d, key=self.d.get, reverse=True)[:self.cut_size]

    @_summary_profile
    def solve_iteratively(self, warm_start=True, method='dual'):
        """Solve the model by feeding in only the top most violated constraints,
        and repeat until no violated constraints remain

        :param warm_start: Set to True to use the previous iteration's optimal
        basis as the initial basis for the current. Intended to be used in
        conjunction with solve_method='dual'
        :param method: 'dual' to solve each iteration with dual simplex,
        'auto' to let gurobi decide which solve method is best

        :return:
        """
        assert isinstance(warm_start, bool)

        self.solve_type = 'iterative'
        self.warm_start = warm_start
        self._instantiate_model(method)

        # Add randomly <self.first_iteration_cuts> of the triangle inequality
        # constraints or just all of them if there's less than <self.first_iteration_cuts>
        self.inf = random.sample(self.c, min(self.first_iteration_cuts, len(self.c)))
        for ((i, j, k), t) in self.inf:
            self._add_triangle_inequality(i, j, k, t)

        self._optimize()
        assert self.mdl.status == gu.GRB.OPTIMAL, 'small initial solve should make solution'

        while True:
            self.d = {}
            for proportion in self.search_proportions:
                self.proportion = proportion
                self._recalibrate_cut_depths()
                self._find_most_violated_constraints()
                if len(self.inf) == self.cut_size:
                    break

            if not self.inf:
                break

            for ((i, j, k), t) in self.inf:
                self._add_triangle_inequality(i, j, k, t)
            self.inf = []

            if not warm_start:
                self.mdl.reset()
            self._optimize()
            assert self.mdl.status == gu.GRB.OPTIMAL, f"model ended up as: {self.mdl.status}"


if __name__ == '__main__':
    from profiler import profile

    @profile(sort_by='cumulative', lines_to_print=10, strip_dirs=True)
    def profilable_iterative():
        mbs = []
        for i in range(10):
            print(f'test {i + 1}')
            mb = MinBisect(n=50, p=.5, q=.1, number_of_cuts=1000)
            mbs.append(mb)
            mb.solve_iteratively()
        return mbs

    @profile(sort_by='cumulative', lines_to_print=10, strip_dirs=True)
    def profilable_once(mbs):
        for i, mb in enumerate(mbs):
            print(f'test {i + 1 + len(mbs)}')
            mb.solve_once(method='auto')

    mbs = profilable_iterative()
    profilable_once(mbs)


    # print run stats
    # solution_schema.csv.write_directory(mb.data, 'test_results', allow_overwrite=True)

