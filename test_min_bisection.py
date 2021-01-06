import gurobipy as gu
from math import isclose
import numpy as np
import os
import random
import re
import unittest
from unittest.mock import patch

from min_bisection import create_constraint_indices, create_adjacency_matrix, \
    MinBisect, solution_schema


class TestCreateAdjacencyMatrix(unittest.TestCase):

    def test_correct_density_within_cluster(self):
        n, p = 100, .5
        a = create_adjacency_matrix(n, p, .1)

        # ratio of actual edge count in cluster1 to max possible edge count in cluster1
        p1 = a[:n//2, :n//2].sum()/((n//2)**2 - n//2)
        self.assertTrue(.45 <= p1 <= .55,
                        msg=(f'a large intracluster density should be near p={p}'))

        # ratio of actual edge count in cluster2 to max possible edge count in cluster2
        p2 = a[n//2:, n//2:].sum()/((n//2)**2 - n//2)
        self.assertTrue(.45 <= p2 <= .55,
                        msg=(f'a large intracluster density should be near p={p}'))

    def test_correct_density_between_clusters(self):
        n, q = 100, .1
        a = create_adjacency_matrix(n, .5, q)

        # ratio of actual edge count between clusters to max possible edge count between clusters
        k = a[:n//2, n//2:].sum()/((n//2)**2 - n//2)
        self.assertTrue(.05 <= k <= .15,
                        msg=(f'large a intercluster density should be near q={q}'))

    def test_reasonable_values(self):
        a = create_adjacency_matrix(10, .5, .1)
        self.assertTrue(set(np.unique(a)) == {0, 1}, 'only values are 0 and 1')
        self.assertTrue(a.trace() == 0, 'no values should exist on diagonal')
        self.assertTrue((a == a.T).all(), 'transpose should be equal')

    def test_correct_dimension(self):
        a = create_adjacency_matrix(10, .5, .1)
        self.assertTrue(a.shape == (10, 10), 'the dimension should be n by n')

    def test_cluster_sizes_correct(self):
        # ensure cluster sizes are correct by forcing all edges within a cluster
        # and none between then counting that the total number in each quadrant correct
        a = create_adjacency_matrix(10, 1, 0)
        self.assertTrue(a[:5, :5].sum() == 20)
        self.assertTrue(a[5:, 5:].sum() == 20)
        self.assertTrue(a[:5, 5:].sum() == 0)
        self.assertTrue(a[5:, :5].sum() == 0)

        a = create_adjacency_matrix(11, 1, 0)
        self.assertTrue(a[:5, :5].sum() == 20)
        self.assertTrue(a[5:, 5:].sum() == 30)
        self.assertTrue(a[:5, 5:].sum() == 0)
        self.assertTrue(a[5:, :5].sum() == 0)


class TestCreateConstraintIndices(unittest.TestCase):
    def test_proper_indices(self):
        indices = range(10)
        c = create_constraint_indices(indices)
        good_idx = set()
        for i in indices:
            for j in indices:
                for k in indices:
                    for t in [1, 2, 3, 4]:
                        if i < j < k:
                            self.assertTrue(((i, j, k), t) in c,
                                            f'i={i}, j={j}, k={k}, t={t} belongs in c')
                            good_idx.add(((i, j, k), t))
        self.assertFalse(c.difference(good_idx),
                         'there should not be any more indices')


class TestMinBisection(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        if os.path.exists('guy_once_auto.txt'):
            os.remove('guy_once_auto.txt')

    def test_init(self):
        # proportion
        mb = MinBisect(8, .5, .1, cut_proportion=.1)
        self.assertTrue(mb.cut_type == 'proportion')
        self.assertTrue(mb.cut_value == .1)

        # fixed
        mb = MinBisect(8, .5, .1, number_of_cuts=10)
        self.assertTrue(mb.cut_type == 'fixed')
        self.assertTrue(mb.cut_value == 10)

        # first iteration cuts
        mb = MinBisect(8, .5, .1, number_of_cuts=10, first_iteration_cuts=100)
        self.assertTrue(mb.first_iteration_cuts == 100)

        # bad inputs
        self.assertRaises(AssertionError, MinBisect, 8, .5, .1)
        self.assertRaises(AssertionError, MinBisect, 8, .5, .1, .1, 10)
        self.assertRaises(AssertionError, MinBisect, 7, .5, .1, .1)
        self.assertRaises(AssertionError, MinBisect, 7, 2, .1, .1)
        self.assertRaises(AssertionError, MinBisect, 7, .5, -1, .1)
        self.assertRaises(AssertionError, MinBisect, 7, .5, .1, 10)
        self.assertRaises(AssertionError, MinBisect, 7, .5, .1, None, .1)
        self.assertRaises(AssertionError, MinBisect, 7, .5, .1, None, 10, 0, 2)
        self.assertRaises(AssertionError, MinBisect, 7, .5, .1, None, 10, 0, .001,
                          'yes')
        self.assertRaises(AssertionError, MinBisect, 7, .5, .1, None, 10, 0, .001,
                          0, '', 'True')
        self.assertRaises(AssertionError, MinBisect, 7, .5, .1, None, 10, 0, .001,
                          0, '', True, -1)

    def test_file_combo(self):
        mb = MinBisect(8, .5, .1, .1)
        mb.solve_type = 'iterative'
        mb.method = 'dual'
        mb.warm_start = True
        self.assertTrue(mb.file_combo == 'iterative_dual_warm')

        mb.solve_type = 'once'
        mb.method = 'auto'
        mb.warm_start = True  # if not iterative, warm_start should be ignored
        self.assertTrue(mb.file_combo == 'once_auto')
        mb.warm_start = False
        self.assertTrue(mb.file_combo == 'once_auto')

        mb.solve_type = 'iterative'
        mb.method = 'auto'
        mb.warm_start = False
        self.assertTrue(mb.file_combo == 'iterative_auto_cold')

    def test_instantiate_model(self):
        indices = range(8)
        mb = MinBisect(8, .5, .1, .1)
        mb.warm_start = True
        mb.solve_type = 'once'
        mb._instantiate_model()
        for i in indices:
            for j in indices:
                if i < j:
                    self.assertTrue((i, j) in mb.x, 'any i < j should be in x')
        mb.mdl.update()
        self.assertTrue(mb.mdl.getConstrByName(f'equal_partitions'),
                        'Equal Partition Constraint should exist')
        self.assertTrue(mb.mdl.getObjective(),
                        'Objective should be set')
        self.assertTrue(mb.mdl.NumVars == mb.n*(mb.n-1)/2,
                        f'we should have {mb.n}*({mb.n}-1)/2 variables')

    def test_instantiate_model_gurobi_parameters(self):
        mb = MinBisect(8, .5, .1, .1, log_to_console=1, log_file_base='guy')
        mb.warm_start = True
        mb.solve_type = 'once'
        mb._instantiate_model('auto')

        self.assertRaises(AssertionError, mb._instantiate_model, 'dog')
        self.assertTrue(mb.method == 'auto')
        self.assertTrue(mb.mdl.params.LogToConsole == 1)
        self.assertTrue(mb.mdl.params.Method == -1)
        self.assertTrue(mb.mdl.params.LogFile == f'guy_{mb.file_combo}.txt')

    def test_instantiate_model_constraints(self):
        mb = MinBisect(8, .5, .1, cut_proportion=.1)
        mb.solve_type = 'iterative'
        mb._instantiate_model()
        self.assertTrue(mb.cut_size == int(.1 * len(mb.c)))

        mb = MinBisect(8, .5, .1, number_of_cuts=10)
        mb.solve_type = 'iterative'
        mb._instantiate_model()
        self.assertTrue(mb.cut_size == 10)

        mb = MinBisect(8, .5, .1, cut_proportion=.1)
        mb.solve_type = 'once'
        mb._instantiate_model()
        self.assertTrue(mb.cut_size == len(mb.c))

    def test_instantiate_model_search_proportions(self):
        # 14 gives us 1456 total constraints which will test that .1 gets added
        # to possible filter sizes to use
        mb = MinBisect(14, .5, .1, cut_proportion=.1)
        mb.solve_type = 'iterative'
        mb._instantiate_model()
        self.assertTrue(mb.search_proportions == [.1, 1])

        mb = MinBisect(14, .5, .1, number_of_cuts=10)
        mb.solve_type = 'iterative'
        mb._instantiate_model()
        self.assertTrue(mb.search_proportions == [.01, .1, 1])

        mb = MinBisect(14, .5, .1, number_of_cuts=10, min_order=1)
        mb.solve_type = 'iterative'
        mb._instantiate_model()
        self.assertTrue(mb.search_proportions == [.1, 1])

        mb = MinBisect(14, .5, .1, number_of_cuts=10, min_order=5)
        mb.solve_type = 'iterative'
        mb._instantiate_model()
        self.assertTrue(mb.search_proportions == [1])

    def test_add_triangle_inequality_adds_constraint_removes_index(self):
        mb = MinBisect(8, .5, .1, .1)
        mb._instantiate_model()
        ((i, j, k), t) = list(mb.c)[0]
        mb._add_triangle_inequality(i, j, k, t)
        mb.mdl.update()

        # test that a constraint is added to model and removed from candidates
        self.assertTrue(mb.mdl.getConstrByName(f'{i}_{j}_{k}_tri{t}'),
                        f'constraint {i}_{j}_{k}_tri{t} should be added')
        self.assertFalse(((i, j, k), t) in mb.c,
                         f"(({i}, {j}, {k}), {t}) should be removed from c")

    def test_add_triangle_inequality_adds_correct_constraint(self):
        for t in range(1, 5):
            mb = MinBisect(8, .5, .1, .1)
            mb._instantiate_model()
            ((i, j, k), t) = [k for k in mb.c if k[1] == t][0]
            mb._add_triangle_inequality(i, j, k, t)
            mb.mdl.update()
            c = mb.mdl.getConstrByName(f'{i}_{j}_{k}_tri{t}')
            lhs, sense, rhs = mb.mdl.getRow(c), c.Sense, c.RHS
            r = re.compile('x_(?P<a>\d)_(?P<b>\d)')
            x = {tuple(int(_) for _ in r.match(lhs.getVar(i).VarName).groups()):
                 lhs.getCoeff(i) for i in range(3)}

            self.assertTrue(sense == '<', 'should be leq constraint')
            self.assertRaises(IndexError, lhs.getVar, 3)
            if t == 1:  # assert adds constraint 1 correctly
                self.assertTrue(x[i, j] == -1)
                self.assertTrue(x[i, k] == -1)
                self.assertTrue(x[j, k] == 1)
                self.assertTrue(rhs == 0)
            elif t == 2:  # assert adds constraint 2 correctly
                self.assertTrue(x[i, j] == -1)
                self.assertTrue(x[i, k] == 1)
                self.assertTrue(x[j, k] == -1)
                self.assertTrue(rhs == 0)
            elif t == 3:  # assert adds constraint 3 correctly
                self.assertTrue(x[i, j] == 1)
                self.assertTrue(x[i, k] == -1)
                self.assertTrue(x[j, k] == -1)
                self.assertTrue(rhs == 0)
            else:  # t == 4, assert adds constraint 4 correctly
                self.assertTrue(x[i, j] == 1)
                self.assertTrue(x[i, k] == 1)
                self.assertTrue(x[j, k] == 1)
                self.assertTrue(rhs == 2)

    def test_add_triangle_inequality_blocks_bad_constraint(self):
        mb = MinBisect(8, .5, .1, .1)
        mb._instantiate_model()
        for t in [0, 5]:
            self.assertRaises(AssertionError, mb._add_triangle_inequality, 1, 2, 3, t)

    def test_summary_profile(self):
        mb = MinBisect(20, .5, .1, .1)
        mb.solve_once('dual')
        self.assertTrue([(0, 'once', 'dual', 'cold')] == list(mb.data.summary_stats.keys()))
        mb.solve_iteratively()
        self.assertTrue([(0, 'once', 'dual', 'cold'), (0, 'iterative', 'dual', 'warm')]
                        == list(mb.data.summary_stats.keys()))
        max_constraints = mb.mdl.NumConstrs
        max_variables = mb.mdl.NumVars

        # make a few runs so run time calcs are correct
        mb.solve_iteratively(warm_start=False)
        mb.solve_once(method='auto')
        mb.solve_iteratively(method='auto')
        mb.solve_iteratively(warm_start=False, method='auto')
        self.assertTrue(solution_schema.good_tic_dat_object(mb.data))

        data = mb.data.summary_stats[0, 'iterative', 'dual', 'warm']
        self.assertTrue(data['n'] == 20)
        self.assertTrue(data['p'] == .5)
        self.assertTrue(data['q'] == .1)
        self.assertTrue(data['cut_type'] == 'proportion')
        self.assertTrue(data['cut_value'] == .1)
        self.assertTrue(data['max_constraints'] == max_constraints)
        self.assertTrue(data['max_variables'] == max_variables)
        self.assertTrue(data['total_cpu_time'] >= data['gurobi_cpu_time'])
        self.assertTrue(data['total_cpu_time'] >= data['non_gurobi_cpu_time'])
        gurobi_cpu_time = sum(d['cpu_time'] for (si, st, m, ws, ssi), d in
                              mb.data.run_stats.items() if st == 'iterative'
                              and m == 'dual' and ws == 'warm')
        self.assertTrue(data['gurobi_cpu_time'] == gurobi_cpu_time)
        self.assertTrue(data['total_cpu_time'] == data['gurobi_cpu_time'] + data['non_gurobi_cpu_time'])
        # compares different runs but they should all be same anyways
        self.assertTrue(isclose(data['objective_value'], mb.mdl.ObjVal, rel_tol=1e-3))

        # make sure all time values > 0
        for f in mb.data.summary_stats.values():
            self.assertTrue(f['gurobi_cpu_time'] >= 0)
            self.assertTrue(f['non_gurobi_cpu_time'] >= 0)

        # make sure all objective values are the same for each run
        objs = [f['objective_value'] for f in mb.data.summary_stats.values()]
        self.assertTrue(all(isclose(obj, objs[0], rel_tol=1e-3) for obj in objs))

        # make sure minimum order and proportions are correctly recorded
        self.assertTrue(data['min_order'] == 0)
        self.assertTrue(data['min_proportion'] == .1)
        data = mb.data.summary_stats[0, 'once', 'auto', 'cold']
        self.assertTrue(data['min_order'] == 0)
        self.assertTrue(data['min_proportion'] == 1)

    def test_optimize(self):
        mb = MinBisect(8, .5, .1, .1, write_mps=True)
        mb.solve_type = 'iterative'
        mb.warm_start = True
        mb._instantiate_model()
        mb._optimize()
        self.assertTrue([(0, 'iterative', 'dual', 'warm', 0)] ==
                        list(mb.data.run_stats.keys()))

        # tests adds a second correctly
        ((i, j, k), t) = [k for k in mb.c][0]
        mb.inf = [((i, j, k), t)]
        mb._add_triangle_inequality(i, j, k, t)
        mb._optimize()
        self.assertTrue([(0, 'iterative', 'dual', 'warm', 0),
                         (0, 'iterative', 'dual', 'warm', 1)]
                        == list(mb.data.run_stats.keys()))

        # tests adds all at once solve correctly
        mb.solve_once(method='dual')
        self.assertTrue([(0, 'iterative', 'dual', 'warm', 0),
                         (0, 'iterative', 'dual', 'warm', 1),
                         (0, 'once', 'dual', 'cold', 0)] ==
                        list(mb.data.run_stats.keys()))
        self.assertTrue(solution_schema.good_tic_dat_object(mb.data))

        # check data filled out as expected
        data = mb.data.run_stats[0, 'iterative', 'dual', 'warm', 1]
        self.assertTrue(data['n'] == 8)
        self.assertTrue(data['p'] == .5)
        self.assertTrue(data['q'] == .1)
        self.assertTrue(data['cut_type'] == 'proportion')
        self.assertTrue(data['cut_value'] == .1)
        self.assertTrue(data['cuts_sought'] == int(.1*224))  # 224 cuts for problem size
        self.assertTrue(data['cuts_added'] == 1)  # only added one manually
        self.assertTrue(data['constraints'] == 2)  # because equal partition and 1 cut
        self.assertTrue(data['variables'] == mb.mdl.NumVars)
        self.assertTrue(data['cpu_time'] >= 0)
        self.assertTrue(data['min_order'] == 0)
        self.assertTrue(data['min_proportion'] == .1)
        self.assertTrue(data['proportion_used'] in [.1, 1])

        data = mb.data.run_stats[0, 'once', 'dual', 'cold', 0]
        self.assertTrue(data['min_order'] == 0)
        self.assertTrue(data['min_proportion'] == 1)
        self.assertTrue(data['proportion_used'] == 1)

        # check mps files created
        for pth in ['model_iterative_dual_warm_0.mps',
                    'model_iterative_dual_warm_1.mps', 'model_once_dual_0.mps']:
            self.assertTrue(os.path.exists(pth))
            os.remove(pth)

    def test_optimize_cut_math_right(self):
        mb = MinBisect(8, .5, .1, number_of_cuts=10)
        mb.first_iteration_cuts = 20
        mb.solve_once(method='dual')
        mb.solve_iteratively()
        d = mb.data.run_stats
        self.assertTrue(d[0, 'iterative', 'dual', 'warm', 0]['cuts_sought'] ==
                        d[0, 'iterative', 'dual', 'warm', 0]['cuts_added'],
                        'cuts sought and added should be same on first iteration')
        self.assertTrue(d[0, 'iterative', 'dual', 'warm', 0]['cuts_sought'] > 10,
                        'cuts first sought should be more than other iterations')
        self.assertTrue(d[0, 'once', 'dual', 'cold', 0]['cuts_sought'] ==
                        d[0, 'once', 'dual', 'cold', 0]['cuts_added'] == 224)
        self.assertTrue(d[0, 'iterative', 'dual', 'warm', 1]['cuts_sought'] == 10)

    def test_solve_once(self):
        a = np.array([[0, 1, 0, 1, 0, 0, 0, 0],
                      [1, 0, 1, 0, 0, 0, 0, 1],
                      [0, 1, 0, 0, 0, 1, 1, 0],
                      [1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 1],
                      [0, 0, 1, 0, 0, 0, 0, 1],
                      [0, 0, 1, 0, 1, 0, 0, 1],
                      [0, 1, 0, 0, 1, 1, 1, 0]])
        mb = MinBisect(8, .8, .1, .1)
        mb.a = a
        mb.solve_once()
        self.assertTrue(mb.solve_type == 'once')
        self.assertFalse(mb.warm_start)
        self.assertTrue(mb.mdl.ObjVal == 3, 'only three edges cross clusters')
        self.assertTrue(mb.x[0, 1].x == mb.x[0, 2].x == mb.x[0, 3].x == 0,
                        '0, 1, 2, and 3 should be one cluster')
        self.assertTrue(mb.x[4, 5].x == mb.x[4, 6].x == mb.x[4, 7].x == 0,
                        '4, 5, 6, and 7 should be one cluster')
        self.assertFalse(mb.c, 'all cuts should have been added')

    def test_solve_iteratively_matches_solve_once_small(self):
        mb = MinBisect(8, .5, .1, .1)
        mb.solve_once()
        once_obj = mb.mdl.ObjVal
        mb.solve_iteratively()
        self.assertTrue(isclose(once_obj, mb.mdl.ObjVal, rel_tol=1e-3),
                        f'one go obj {once_obj} but iterative obj {mb.mdl.ObjVal}')
        self.assertTrue(mb.solve_type == 'iterative')
        mb.solve_iteratively(warm_start=False)
        self.assertFalse(mb.warm_start)
        self.assertRaises(AssertionError, mb.solve_iteratively, 'True')

    def test_solve_iteratively_matches_solve_once_big(self):
        mb = MinBisect(40, .5, .1, .1)
        mb.solve_once()
        once_obj = mb.mdl.ObjVal
        mb.solve_iteratively()
        self.assertTrue(isclose(once_obj, mb.mdl.ObjVal, rel_tol=1e-3),
                        f'one go obj {once_obj} but iterative obj {mb.mdl.ObjVal}')

    def test_recalibrate_cut_depths(self):
        a = np.array([[0, 1, 0, 1],
                      [1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [1, 0, 1, 0]])
        mb = MinBisect(4, .8, .5, .1)
        mb.a = a

        mb.solve_type = 'iterative'
        # filter no cuts to ensure calculation right
        mb.tolerance = -1000
        mb._instantiate_model()
        mb.mdl.optimize()
        mb._recalibrate_cut_depths()  # just makes sure that the cut calc is right
        for (i, j, k) in {(i, j, k) for ((i, j, k), t) in mb.c}:
            self.assertTrue(mb.d[(i, j, k), 1] == mb.x[j, k].x - mb.x[i, j].x - mb.x[i, k].x)
            self.assertTrue(mb.d[(i, j, k), 2] == mb.x[i, k].x - mb.x[i, j].x - mb.x[j, k].x)
            self.assertTrue(mb.d[(i, j, k), 3] == mb.x[i, j].x - mb.x[i, k].x - mb.x[j, k].x)
            self.assertTrue(mb.d[(i, j, k), 4] == mb.x[i, j].x + mb.x[i, k].x + mb.x[j, k].x - 2)

        tol = .01
        mb.tolerance = tol
        inf_cuts = {idx for idx, depth in mb.d.items() if depth > tol}
        mb.d = {}
        mb._recalibrate_cut_depths()
        self.assertFalse(inf_cuts.difference(mb.d.keys()))
        self.assertTrue(len(mb.d) == 2)

    def test_find_most_violated_constraints(self):
        # use 100 to ensure we get some values less than 1 to test that sort is right
        mb = MinBisect(52, .5, .1, number_of_cuts=100)
        mb.solve_type = 'iterative'
        mb._instantiate_model()
        mb.first_iteration_cuts = 5000
        mb.inf = random.sample(mb.c, min(mb.first_iteration_cuts, len(mb.c)))
        for ((i, j, k), t) in mb.inf:
            mb._add_triangle_inequality(i, j, k, t)
        mb._optimize()
        mb._recalibrate_cut_depths()
        mb._find_most_violated_constraints()
        self.assertTrue(len(mb.inf) == 100, 'we only add 100 cuts at once')
        for k in random.sample(mb.inf, 20):
            self.assertTrue(len([_ for _ in mb.d if mb.d[_] > mb.d[k]]) < 100,
                            'there should be fewer than 100 constraints more violated')

    def test_find_most_violated_constraints_when_it_selects_all(self):
        mb = MinBisect(20, .5, .1, number_of_cuts=5000)
        mb.solve_type = 'iterative'
        mb._instantiate_model()
        mb._optimize()
        mb._recalibrate_cut_depths()
        mb._find_most_violated_constraints()
        # when using only cardinality constraint, 1 is only positive cut value
        self.assertTrue(len(mb.inf) == len([k for k in mb.d if mb.d[k] == 1]),
                        'select all infeasible cuts when total less than sought')

    def test_recalibrate_cut_depths_uses_tolerance(self):
        mb = MinBisect(4, .5, .1, number_of_cuts=20)
        mb._instantiate_model()
        mb.mdl.update()
        for (i, j), var in mb.x.items():
            var.lb = -1
            mb.mdl.addConstr(mb.x[i, j] == -.00001)
        mb.mdl.remove(mb.mdl.getConstrByName('equal_partitions'))
        mb.mdl.optimize()

        mb._recalibrate_cut_depths()
        mb._find_most_violated_constraints()
        self.assertTrue(len(mb.inf) == 0,
                        'all cut depths should be < tolerance')

        mb.tolerance = .000001
        mb._recalibrate_cut_depths()
        mb._find_most_violated_constraints()
        self.assertTrue(len(mb.inf) == 12,
                        'only last triangle inequalities should be violated')

    @patch('min_bisection.gu.Model.reset')
    def test_solve_iteratively_cold(self, reset_patch):
        mb = MinBisect(8, .5, .1, number_of_cuts=20)
        mb.solve_iteratively(warm_start=False)
        # reset calls should equal number of iterations after the 1st
        self.assertTrue(reset_patch.call_count == len(mb.data.run_stats) - 1)

    @patch('min_bisection.gu.Model.reset')
    def test_solve_iteratively_warm(self, reset_patch):
        mb = MinBisect(20, .5, .1, number_of_cuts=20)
        mb.solve_iteratively()
        self.assertFalse(reset_patch.called)


if __name__ == '__main__':
    unittest.main()