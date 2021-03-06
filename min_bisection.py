import gurobipy as gu
from math import ceil, floor
import numpy as np
from profiler import profile_run_time, profile_memory
import random
import re
from ticdat import TicDatFactory
import time

# headers for our output files
solution_schema = TicDatFactory(
    run_stats=[['solve_id', 'solve_type', 'method', 'warm_start',
                'min_search_proportion', 'threshold_proportion', 'remove_constraints',
                'zero_slack_likelihood', 'sub_solve_id'],
               ['n', 'p', 'q', 'cut_type', 'cut_value', 'cuts_sought',
                'cuts_added', 'cuts_removed', 'search_proportion_used',
                'current_threshold', 'variables',
                'constraints', 'cpu_time', 'dual_0_constraints', 'dual_0_with_slack',
                'dual_0_no_slack']],
    summary_stats=[['solve_id', 'solve_type', 'method', 'warm_start',
                    'min_search_proportion', 'threshold_proportion', 'remove_constraints',
                    'zero_slack_likelihood'],
                   ['n', 'p', 'q', 'cut_type', 'cut_value', 'max_variables',
                    'max_constraints', 'total_cpu_time', 'gurobi_cpu_time',
                    'non_gurobi_cpu_time', 'objective_value', 'dual_0_constraints',
                    'dual_0_with_slack', 'dual_0_no_slack']]
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
                 write_mps=False, first_iteration_cuts=100):
        """Create our adjacency matrix and declare all other needed attributes

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

        self.n = n
        self.p = p
        self.q = q
        self.solve_id = solve_id
        self.tolerance = tolerance
        self.log_to_console = log_to_console
        self.log_file_base = log_file_base
        self.write_mps = write_mps
        self.first_iteration_cuts = first_iteration_cuts

        self.indices = range(n)
        self.a = create_adjacency_matrix(n, p, q)
        self.c = None  # available cuts
        self.d = {}  # cut depths
        self.v = {}
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
        self.method = None
        self.removed = []
        self.pattern = re.compile(r'^(\d+)_(\d+)_(\d+)_tri(\d)$')

        # to be assigned at _instantiate_model
        self.warm_start = None
        self.search_proportions = None
        self.current_search_proportion = None
        self.min_search_proportion = None
        self.threshold_proportion = None
        self.current_threshold = None
        self.keep_iterating = None
        self.remove_constraints = False
        self.removed_nonslack = None
        self.zero_slack_likelihood = None
        self.dual_0_constraints = None
        self.dual_0_with_slack = None
        self.dual_0_no_slack = None


    @property
    def file_combo(self):
        string = f'_{self.warm_str}_{self.min_search_proportion}_{self.threshold_proportion}'
        return f'{self.solve_type}_{self.method}' \
               f'{"" if self.solve_type != "iterative" else string}'

    @property
    def warm_str(self):
        return 'warm' if self.warm_start else 'cold'

    def _reset_dual_0_counters(self):
        self.dual_0_constraints = 0
        self.dual_0_with_slack = 0
        self.dual_0_no_slack = 0

    def _instantiate_model(self, solve_type='iterative', warm_start=True,
                           method='dual', min_search_proportion=1,
                           threshold_proportion=None, remove_constraints=False,
                           zero_slack_likelihood=0):
        """Does everything that solving iteratively and at once will share, e.g.
        instantiating the model object and variables as well as setting the
        objective and equal partition constraint.

        :param solve_type: whether this is an 'iterative' or 'once' solve
        :param warm_start: Set to True to use the previous iteration's optimal
        basis as the initial basis for the current. Intended to be used in
        conjunction with solve_method='dual'
        :param method: 'dual' to solve each iteration with dual simplex,
        'auto' to let gurobi decide which solve method is best
        :param min_search_proportion: smallest proportion of available cuts to search
        for deepest cuts to add to the model when solving iteratively. Note,
        search_proportions are inverse powers of 10, so this parameter will be rounded
        up to the next largest inverse power of 10. For example, .02 becomes .1.
        :param threshold_proportion: if in (0, 1), what proportion of current infeasible
        cuts must a cut be deeper than to be added to the model when solved iteratively
        :param remove_constraints: whether or not to remove constraints deemed
        as not useful for solving subsequent iterations
        :param zero_slack_likelihood: likelihood we remove a constraint with a
        dual value of 0 but also 0 slack.

        Note: it is intended for the user to use at most one of min_search_proportion
        and threshold_proportion as they are not designed to be used at the same time.

        :return:
        """
        # solve specific MinBisect instantiation
        assert solve_type in ['iterative', 'once']
        assert isinstance(warm_start, bool)
        assert method in ['dual', 'auto'], 'solve with dual or let gurobi decide'
        assert 0 < min_search_proportion <= 1
        assert threshold_proportion is None or (0 < threshold_proportion < 1)
        assert min_search_proportion == 1 or threshold_proportion is None, 'not both'
        assert isinstance(remove_constraints, bool)
        assert 0 <= zero_slack_likelihood <= 1

        self.solve_type = solve_type
        self.warm_start = warm_start
        self.min_search_proportion = min_search_proportion
        self.threshold_proportion = threshold_proportion
        self.method = method
        self.remove_constraints = remove_constraints
        self.zero_slack_likelihood = zero_slack_likelihood
        self.c = create_constraint_indices(self.indices)
        self.sub_solve_id = -1
        self.cut_size = len(self.c) if self.solve_type == 'once' else \
            floor(self.cut_value * len(self.c)) if self.cut_type == 'proportion' \
            else self.cut_value
        self.search_proportions = [
            10**x for x in range(ceil(np.log10(self.cut_size/len(self.c))), 0)
            if 10**x >= self.min_search_proportion
        ] + [1]
        self.current_search_proportion = 1
        self.current_threshold = None
        self.keep_iterating = True
        self.removed_nonslack = set()
        self._reset_dual_0_counters()

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
        :param t: whether this is constraint type 1, 2, 3, 4
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
        """A decorator that is used to collect metadata on once solves
        and iterative solves

        :return:
        """
        def wrapper(self, *args, **kwargs):
            solve_start = time.process_time()
            retval = func(self, *args, **kwargs)
            total_cpu_time = time.process_time() - solve_start

            def sum_column(column):
                """ sum run stat column over sub solve indices for this pk combination

                :param column: which column to take a sum of
                :return:
                """
                return sum(
                    d[column] for (si, st, m, ws, msp, tp, rc, zsl, ssi), d in
                    self.data.run_stats.items() if self.solve_type == st and
                    self.method == m and self.warm_str == ws and
                    self.min_search_proportion == msp and self.threshold_proportion == tp
                    and self.remove_constraints == rc and self.zero_slack_likelihood == zsl
                )

            gurobi_cpu_time = sum_column('cpu_time')
            self.data.summary_stats[self.solve_id, self.solve_type, self.method,
                                    self.warm_str, self.min_search_proportion,
                                    self.threshold_proportion, self.remove_constraints,
                                    self.zero_slack_likelihood] = {
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
                'objective_value': self.mdl.ObjVal,
                'dual_0_constraints': sum_column('dual_0_constraints'),
                'dual_0_with_slack': sum_column('dual_0_with_slack'),
                'dual_0_no_slack': sum_column('dual_0_no_slack')
            }
            return retval

        return wrapper

    def _optimize(self):
        """ A function that collects data on each LP solve

        :return:
        """
        self.sub_solve_id += 1
        if self.write_mps:
            name = f'model_{self.file_combo}_{self.sub_solve_id}'
            self.mdl.write(f'{name}.mps')
            if self.sub_solve_id > 0:
                self.mdl.write(f'{name}.bas')
        sub_solve_start = time.process_time()
        self.mdl.optimize()
        sub_solve_cpu_time = time.process_time() - sub_solve_start
        self.constraints = self.mdl.NumConstrs
        self.variables = self.mdl.NumVars
        self.data.run_stats[self.solve_id, self.solve_type, self.method,
                            self.warm_str, self.min_search_proportion,
                            self.threshold_proportion, self.remove_constraints,
                            self.zero_slack_likelihood, self.sub_solve_id] = {
            'n': self.n,
            'p': self.p,
            'q': self.q,
            'cut_type': self.cut_type,
            'cut_value': self.cut_value,
            'cuts_sought': len(self.inf) if self.solve_type == 'iterative' and
                self.sub_solve_id == 0 else self.cut_size,
            'cuts_added': len(self.inf) if self.solve_type == 'iterative' else self.cut_size,
            'cuts_removed': len(self.removed),
            'search_proportion_used': self.current_search_proportion,
            'current_threshold': self.current_threshold,
            'constraints': self.mdl.NumConstrs,
            'variables': self.mdl.NumVars,
            'cpu_time': sub_solve_cpu_time,
            'dual_0_constraints': self.dual_0_constraints,
            'dual_0_with_slack': self.dual_0_with_slack,
            'dual_0_no_slack': self.dual_0_no_slack
        }

    @_summary_profile
    def solve_once(self, method='auto', run_time_profile_file=None, memory_profile_file=None):
        """Solves the model with all constraints added at once.
         
         :param run_time_profile_file: name to give runtime profiler output if its activated
         :param memory_profile_file: name to give memory profiler output if its activated
         
        For additional parameter explanation, see MinBisect._instantiate_model

        :return:
        """
        self._instantiate_model('once', False, method)

        # make new object so loop not thrown off by deletion
        for ((i, j, k), t) in list(self.c):
            self._add_triangle_inequality(i, j, k, t)

        self._optimize()
        assert self.mdl.status == gu.GRB.OPTIMAL, 'once solve should have solution'

    def _get_cut_depth(self, i, j, k, t):
        """find how much each constraint is violated. don't worry about
        normalizing since each vector has the same norm

        :param i: ith index
        :param j: jth index
        :param k: kth index
        :param t: what type of constraint is this
        :return: depth of the cut
        """

        if t == 1:
            try:
                return self.v[j, k] - self.v[i, j] - self.v[i, k]
            except KeyError:
                self._get_vals(i, j, k)
                return self.v[j, k] - self.v[i, j] - self.v[i, k]
        elif t == 2:
            try:
                return self.v[i, k] - self.v[i, j] - self.v[j, k]
            except KeyError:
                self._get_vals(i, j, k)
                return self.v[i, k] - self.v[i, j] - self.v[j, k]
        elif t == 3:
            try:
                return self.v[i, j] - self.v[i, k] - self.v[j, k]
            except KeyError:
                self._get_vals(i, j, k)
                return self.v[i, j] - self.v[i, k] - self.v[j, k]
        else:  # t == 4:
            try:
                return self.v[i, j] + self.v[i, k] + self.v[j, k] - 2
            except KeyError:
                self._get_vals(i, j, k)
                return self.v[i, j] + self.v[i, k] + self.v[j, k] - 2

    def _get_vals(self, i, j, k):
        for (a, b) in [(j, k), (i, j), (i, k)]:
            if (a, b) not in self.v:
                self.v[a, b] = self.x[a, b].x

    def _recalibrate_cut_depths_by_threshold_proportion(self):
        """Find the first <self.cut_size> unsatisfied constraints that are
        violated by more than <self.current_threshold>, stopping once they have
        been found. Returns all constraints violated by more than <self.current_threshold>
        if less than <self.cut_size> such constraints are found

        for a constraint to be satisfied, its key should have a corresponding
        value <= 0. Changing this such that the value <= tolerance allows constraints
        close to being satisfied to still be considered satisfied

        :return:
        """
        count = 0
        for ((i, j, k), t) in self.c:
            cut_depth = self._get_cut_depth(i, j, k, t)
            if cut_depth >= self.current_threshold:
                self.d[(i, j, k), t] = cut_depth
                count += 1
                if count == self.cut_size:
                    break

    def _recalibrate_cut_depths_by_search_proportion(self):
        """From a random subset of <self.search_proportion> of the total
        remaining constraints, find the <self.cut_size> most unsatisfied that
        are unsatisfied by more than the tolerance.

        :return:
        """
        for ((i, j, k), t) in self.c if self.current_search_proportion == 1 else \
                random.sample(self.c, int(self.current_search_proportion * len(self.c))):
            cut_depth = self._get_cut_depth(i, j, k, t)
            if cut_depth > self.tolerance:
                self.d[(i, j, k), t] = cut_depth

    def _find_new_threshold(self):
        """ Find which unadded constraint has a depth of violation that drops it
        in the 100*<self.threshold_proportion>th percentile of violation depths
        of all constraints

        :return:
        """
        key = sorted(self.d, key=self.d.get)[floor(len(self.d) * self.threshold_proportion)]
        self.current_threshold = self.d[key]

    def _find_most_violated_constraints(self):
        """Find all constraint indices that represent infeasible constraints (i.e.
        have values greater than the tolerance) then take the <self.cut_size>
        most violated. Optionally, use a heuristic to identify these most violated
        from only a subset of all unadded constraints

        :return:
        """
        self.d = {}
        self.v = {}
        if self.current_threshold:
            self._recalibrate_cut_depths_by_threshold_proportion()
            self.inf = list(self.d.keys())
            if len(self.inf) < self.cut_size:
                self.current_threshold = None
        else:
            for search_proportion in self.search_proportions:
                self.current_search_proportion = search_proportion
                self._recalibrate_cut_depths_by_search_proportion()
                self.inf = sorted(self.d, key=self.d.get, reverse=True)[:self.cut_size]
                # only find new threshold if there are at least <cut_size>
                # constraints with that depth or more
                if self.threshold_proportion and \
                        self.cut_size < len(self.d) * (1 - self.threshold_proportion):
                    self._find_new_threshold()
                if len(self.inf) == self.cut_size:
                    break
            if not self.inf:
                self.keep_iterating = False

    def _remove_constraints(self):
        """Removes all constraints from the model which have no reduced cost,
        i.e. removing the constraint will not change the optimal solution

        :return removed: a list of the removed constraints
        """
        removed = []
        for constr in self.mdl.getConstrs():
            # ignore if not an added cut with ~0 dual value
            if constr.ConstrName == 'equal_partitions' or constr.pi <= -1e-10:
                continue
            i, j, k, t = [int(idx) for idx in
                          self.pattern.match(constr.ConstrName).groups()]
            self.dual_0_constraints += 1
            dual_0_removable = ((i, j, k), t) not in self.removed_nonslack and \
                np.random.uniform() < self.zero_slack_likelihood
            if constr.slack <= self.tolerance:
                self.dual_0_no_slack += 1
            # remove if there is slack or if this is a zero dual that can go
            if constr.slack > self.tolerance or dual_0_removable:
                if constr.slack > self.tolerance:
                    self.dual_0_with_slack += 1
                else:
                    self.removed_nonslack.add(((i, j, k), t))
                removed.append(((i, j, k), t))
                self.mdl.remove(constr)
        return removed

    def _iterate(self):
        """Complete one iteration of an iterative solve. This method includes:
        * removing constraints deemed not needed
        * adding the most violated outstanding constraints
        * resolving the new model if new constraints added

        :return:
        """
        self._reset_dual_0_counters()
        if self.remove_constraints:
            self.removed = self._remove_constraints()
        self._find_most_violated_constraints()
        if not self.keep_iterating:
            return
        self.c.update(self.removed)  # add removed constraints back to potential cuts
        for ((i, j, k), t) in self.inf:
            self._add_triangle_inequality(i, j, k, t)
        if not self.warm_start:
            self.mdl.reset()
        self._optimize()
        assert self.mdl.status == gu.GRB.OPTIMAL, f"model ended up as: {self.mdl.status}"

    @_summary_profile
    @profile_run_time(sort_by='tottime', lines_to_print=20, strip_dirs=True)
    def solve_iteratively(self, warm_start=True, method='dual',
                          min_search_proportion=1, threshold_proportion=None,
                          remove_constraints=False, zero_slack_likelihood=0,
                          run_time_profile_file=None, memory_profile_file=None):
        """Solve the model by feeding in only the top most violated constraints,
        and repeat until no violated constraints remain.
        
        :param run_time_profile_file: name to give runtime profiler output if its activated
        :param memory_profile_file: name to give memory profiler output if its activated
        
        For explanation of the parameters, see MinBisect._instantiate_model

        :return:
        """
        self._instantiate_model('iterative', warm_start, method,
                                min_search_proportion, threshold_proportion,
                                remove_constraints, zero_slack_likelihood)

        # Add randomly <self.first_iteration_cuts> of the triangle inequality
        # constraints or just all of them if there's less than <self.first_iteration_cuts>
        self.inf = random.sample(self.c, min(self.first_iteration_cuts, len(self.c)))
        for ((i, j, k), t) in self.inf:
            self._add_triangle_inequality(i, j, k, t)

        self._optimize()
        assert self.mdl.status == gu.GRB.OPTIMAL, 'small initial solve should make solution'

        while self.keep_iterating:
            self._iterate()


# @profile_run_time(sort_by='tottime', lines_to_print=10, strip_dirs=True)
def removed(x):
    mbs = []
    for i in range(x):
        print(f'test {i + 1}')
        mb = MinBisect(n=40, p=.5, q=.2, number_of_cuts=1000)
        mbs.append(mb)
        mb.solve_iteratively(method='auto', remove_constraints=True)
    return mbs


# @profile_run_time(sort_by='tottime', lines_to_print=10, strip_dirs=True)
def non_removed(mbs):
    for i, mb in enumerate(mbs):
        print(f'test {i + 1 + len(mbs)}')
        mb.solve_iteratively(method='auto')


if __name__ == '__main__':

    mbs = removed(1)
    non_removed(mbs)

    # print run stats
    # solution_schema.csv.write_directory(mb.data, 'test_results', allow_overwrite=True)
