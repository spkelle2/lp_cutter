import gurobipy as gu
import numpy as np
import random
import sys
from ticdat import TicDatFactory
import time

solution_schema = TicDatFactory(
    run_stats=[['solve_id', 'solve_type', 'sub_solve_id'],
               ['n', 'p', 'q', 'cut_type', 'cut_value', 'cuts_sought',
                'cuts_added', 'variables', 'constraints', 'cpu_time']],
    summary_stats=[['solve_id', 'solve_type'],
                   ['n', 'p', 'q', 'cut_type', 'cut_value', 'max_variables',
                    'max_constraints', 'total_cpu_time', 'gurobi_cpu_time',
                    'non_gurobi_cpu_time', 'objective_value']]
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

    c = {((i, j, k), idx): 0 for i in indices for j in indices[i + 1:]
         for k in indices[j + 1:] for idx in [1, 2, 3, 4]}
    return c


class MinBisect:

    def __init__(self, n, p, q, cut_proportion=None, number_of_cuts=None,
                 solve_id=0, tolerance=.0001, output_flag=0):
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
        :param output_flag: 0 to run gurobi without printing outputs, 1 otherwise
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
        assert output_flag in [0, 1], 'gurobi requires output flag to either be 0 or 1'

        self.n = n
        self.p = p
        self.q = q
        self.solve_id = solve_id
        self.tolerance = tolerance
        self.output_flag = output_flag
        self.indices = range(n)
        self.a = create_adjacency_matrix(n, p, q)
        self.c = None
        self.cut_type = 'proportion' if cut_proportion else 'fixed'
        self.cut_value = cut_proportion if cut_proportion else number_of_cuts
        self.cut_size = None
        self.mdl = None
        self.x = None
        self.current_sub_solve_id = -1
        self.variables = 0
        self.constraints = 0
        self.data = solution_schema.TicDat()
        self.solve_type = None
        self.inf = []
        self.first_iteration_cuts = 100

    def _instantiate_model(self):
        """Does everything that solving iteratively and at once will share, namely
        instantiating the model and variables as well as setting the objective
        and equal partition constraint.

        :return:
        """
        self.c = create_constraint_indices(self.indices)
        self.cut_size = len(self.c) if self.solve_type == 'once' else \
            int(self.cut_value * len(self.c)) if self.cut_type == 'proportion' else \
                self.cut_value
        self.mdl = gu.Model("min bisection")  # check to make sure this gives empty model
        self.mdl.setParam(gu.GRB.Param.OutputFlag, self.output_flag)
        self.mdl.setParam(gu.GRB.Param.Method, 1)

        # variables
        self.x = {(i, j): self.mdl.addVar(ub=1, name=f'x_{i}_{j}') for i in self.indices
                  for j in self.indices if i < j}

        # objective
        self.mdl.setObjective(gu.quicksum(self.a[i, j] * self.x[i, j] for (i, j)
                                          in self.x), sense=gu.GRB.MINIMIZE)

        # (3) Equal partition constraint
        self.mdl.addConstr(gu.quicksum(self.x[i, j] for (i, j) in self.x)
                           == self.n ** 2 / 4, name='Equal Partitions')

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
        del self.c[(i, j, k), t]

    def _summary_profile(func):
        def wrapper(self):
            solve_start = time.process_time()
            retval = func(self)
            total_cpu_time = time.process_time() - solve_start
            gurobi_cpu_time = sum(d['cpu_time'] for (_, solve_type, _), d in
                                  self.data.run_stats.items() if
                                  self.solve_type == solve_type)
            self.data.summary_stats[self.solve_id, self.solve_type] = {
                'n': self.n,
                'p': self.p,
                'q': self.q,
                'cut_type': self.cut_type,
                'cut_value': self.cut_value,
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
        if self.solve_type != 'once':
            self.current_sub_solve_id += 1
        sub_solve_start = time.process_time()
        self.mdl.optimize()
        sub_solve_cpu_time = time.process_time() - sub_solve_start
        self.constraints = self.mdl.NumConstrs
        self.variables = self.mdl.NumVars
        sub_solve_id = 0 if self.solve_type == 'once' else self.current_sub_solve_id
        self.data.run_stats[self.solve_id, self.solve_type, sub_solve_id] = {
            'n': self.n,
            'p': self.p,
            'q': self.q,
            'cut_type': self.cut_type,
            'cut_value': self.cut_value,
            'cuts_sought': len(self.inf) if self.solve_type == 'iterative' and
                                            sub_solve_id == 0 else self.cut_size,
            'cuts_added': len(self.inf) if self.solve_type == 'iterative' else self.cut_size,
            'constraints': self.mdl.NumConstrs,
            'variables': self.mdl.NumVars,
            'cpu_time': sub_solve_cpu_time
        }

    @_summary_profile
    def solve_once(self):
        """Solves the model with all constraints added at once

        :return:
        """
        self.solve_type = 'once'
        self._instantiate_model()
        keys = list(self.c.keys())
        for ((i, j, k), t) in keys:  # may need to make a list first
            self._add_triangle_inequality(i, j, k, t)

        self._optimize()
        assert self.mdl.status == gu.GRB.OPTIMAL, 'small initial solve should make solution'

    def _recalibrate_cut_depths(self):
        """find how much each constraint is violated. don't worry about
        normalizing since each vector has the same norm

        for a constraint to be satisfied, its key should have a corresponding
        value <= 0. Changing this such that the value <= tolerance allows constraints
        close to being satisfied to still be considered satisfied

        :return:
        """
        for ((i, j, k), t) in self.c:
            if t == 1:
                self.c[(i, j, k), t] = self.x[j, k].x - self.x[i, j].x - self.x[i, k].x
            elif t == 2:
                self.c[(i, j, k), t] = self.x[i, k].x - self.x[i, j].x - self.x[j, k].x
            elif t == 3:
                self.c[(i, j, k), t] = self.x[i, j].x - self.x[i, k].x - self.x[j, k].x
            else:  # t == 4:
                self.c[(i, j, k), t] = self.x[i, j].x + self.x[i, k].x + self.x[j, k].x - 2

    @_summary_profile
    def solve_iteratively(self):
        """Solve the model by feeding in only the top most violated constraints,
        and repeat until no violated constraints remain

        :return:
        """
        self.solve_type = 'iterative'
        self._instantiate_model()
        # Add randomly <self.first_iteration_cuts> of the triangle inequality
        # constraints or just all of them if there's less than <self.first_iteration_cuts>
        self.inf = random.sample(self.c.keys(), min(self.first_iteration_cuts,
                                                    len(self.c)))
        for ((i, j, k), t) in self.inf:
            self._add_triangle_inequality(i, j, k, t)

        self._optimize()
        assert self.mdl.status == gu.GRB.OPTIMAL, 'small initial solve should make solution'

        while True:
            self._recalibrate_cut_depths()
            self.inf = [k for k in sorted(self.c, key=self.c.get, reverse=True) if
                        self.c[k] > self.tolerance][:self.cut_size]
            if not self.inf:
                break

            for ((i, j, k), t) in self.inf:
                self._add_triangle_inequality(i, j, k, t)

            self._optimize()
            assert self.mdl.status == gu.GRB.OPTIMAL, f"model ended up as: {self.mdl.status}"


if __name__ == '__main__':
    start = time.time()
    mb = MinBisect(n=int(sys.argv[1]), p=float(sys.argv[2]), q=float(sys.argv[3]),
                   cut_proportion=float(sys.argv[4]))
    mb.solve_iteratively()
    print(f'solve time: {time.time() - start} seconds')
