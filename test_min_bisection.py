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
from slim_min_bisection import solve_iterative_min_bisect


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
        for item in [_ for _ in os.listdir() if _.endswith('.prof')]:
            os.remove(item)
        if os.path.exists('guy_iterative_auto_warm_1_None.txt'):
            os.remove('guy_iterative_auto_warm_1_None.txt')

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

        # finding inactiave constraints
        self.assertTrue(0 <= mb.act_tol < 1)
        i, j, k, t = [int(_) for _ in mb.pattern.match('1_2_3_tri2').groups()]
        self.assertTrue(i == 1 and j == 2 and k == 3 and t == 2)

        # bad inputs
        self.assertRaises(AssertionError, MinBisect, 8, .5, .1)
        self.assertRaises(AssertionError, MinBisect, 8, .5, .1, .1, 10)
        self.assertRaises(AssertionError, MinBisect, 7, .5, .1, .1)
        self.assertRaises(AssertionError, MinBisect, 8, 2, .1, .1)
        self.assertRaises(AssertionError, MinBisect, 8, .5, -1, .1)
        self.assertRaises(AssertionError, MinBisect, 8, .5, .1, 10)
        self.assertRaises(AssertionError, MinBisect, 8, .5, .1, None, .1)
        self.assertRaises(AssertionError, MinBisect, 8, .5, .1, None, 10, 0, 2)
        self.assertRaises(AssertionError, MinBisect, 8, .5, .1, None, 10, 0, .001,
                          'yes')
        self.assertRaises(AssertionError, MinBisect, 8, .5, .1, None, 10, 0, .001,
                          0, '', 'True')
        self.assertRaises(AssertionError, MinBisect, 8, .5, .1, None, 10, 0, .001,
                          0, '', True, -1)

    def test_file_combo(self):
        mb = MinBisect(8, .5, .1, .1)
        mb.solve_type = 'iterative'
        mb.method = 'dual'
        mb.warm_start = True
        mb.min_search_proportion = 1
        mb.threshold_proportion = None
        self.assertTrue(mb.file_combo == 'iterative_dual_warm_1_None')

        mb.solve_type = 'once'
        mb.method = 'auto'
        mb.warm_start = True  # if not iterative, warm_start should be ignored
        self.assertTrue(mb.file_combo == 'once_auto')
        mb.warm_start = False
        self.assertTrue(mb.file_combo == 'once_auto')

        mb.solve_type = 'iterative'
        mb.method = 'auto'
        mb.warm_start = False
        self.assertTrue(mb.file_combo == 'iterative_auto_cold_1_None')

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
        mb._instantiate_model(method='auto')

        self.assertRaises(AssertionError, mb._instantiate_model, 'dog')
        self.assertTrue(mb.method == 'auto')
        self.assertTrue(mb.mdl.params.LogToConsole == 1)
        self.assertTrue(mb.mdl.params.Method == -1)
        self.assertTrue(mb.mdl.params.LogFile == f'guy_{mb.file_combo}.txt')

    def test_instantiate_model_cut_size(self):
        mb = MinBisect(8, .5, .1, cut_proportion=.1)
        mb._instantiate_model('iterative')
        self.assertTrue(mb.cut_size == int(.1 * len(mb.c)))

        mb = MinBisect(8, .5, .1, number_of_cuts=10)
        mb._instantiate_model('iterative')
        self.assertTrue(mb.cut_size == 10)

        mb = MinBisect(8, .5, .1, cut_proportion=.1)
        mb._instantiate_model('once')
        self.assertTrue(mb.cut_size == len(mb.c))

    def test_instantiate_model_search_proportions(self):
        # 14 gives us 1456 total constraints which will test that .1 gets added
        # to possible filter sizes to use
        mb = MinBisect(14, .5, .1, cut_proportion=.1)
        mb._instantiate_model('iterative', min_search_proportion=.001)
        self.assertTrue(mb.search_proportions == [.1, 1])

        mb = MinBisect(14, .5, .1, number_of_cuts=10)
        mb._instantiate_model('iterative', min_search_proportion=.001)
        self.assertTrue(mb.search_proportions == [.01, .1, 1],
                        '.001 shouldnt be selected since 10*1000 > 1456')

        mb._instantiate_model('iterative', min_search_proportion=.1)
        self.assertTrue(mb.search_proportions == [.1, 1])

        mb._instantiate_model('iterative')
        self.assertTrue(mb.search_proportions == [1])

    def test_instantiate_model_sets_up_min_bisect(self):
        mb = MinBisect(8, .5, .1, number_of_cuts=100)
        mb._instantiate_model(solve_type='iterative', warm_start=True, method='dual',
                              min_search_proportion=1, threshold_proportion=None,
                              act_tol=.1)

        self.assertTrue(mb.solve_type == 'iterative')
        self.assertTrue(mb.warm_start)
        self.assertTrue(mb.method == 'dual')
        self.assertTrue(mb.min_search_proportion == 1)
        self.assertTrue(mb.threshold_proportion is None)
        self.assertIsInstance(mb.c, set)
        self.assertTrue(mb.sub_solve_id == -1)
        self.assertTrue(mb.current_search_proportion == 1)
        self.assertTrue(mb.current_threshold is None)
        self.assertTrue(mb.keep_iterating)
        self.assertTrue(mb.act_tol)

    def test_instantiate_model_passes_asserts(self):
        mb = MinBisect(8, .5, .1, number_of_cuts=100)
        self.assertRaises(AssertionError, mb._instantiate_model,
                          solve_type='something_else', warm_start=True, method='dual',
                          min_search_proportion=1, threshold_proportion=None)
        self.assertRaises(AssertionError, mb._instantiate_model,
                          solve_type='iterative', warm_start='True', method='dual',
                          min_search_proportion=1, threshold_proportion=None)
        self.assertRaises(AssertionError, mb._instantiate_model,
                          solve_type='iterative', warm_start=True, method='dual',
                          min_search_proportion=2, threshold_proportion=None)
        self.assertRaises(AssertionError, mb._instantiate_model,
                          solve_type='iterative', warm_start=True, method='dual',
                          min_search_proportion=1, threshold_proportion=2)
        self.assertRaises(AssertionError, mb._instantiate_model,
                          solve_type='iterative', warm_start=True, method='dual',
                          min_search_proportion=.5, threshold_proportion=.5)
        self.assertRaises(AssertionError, mb._instantiate_model,
                          solve_type='iterative', warm_start=True, method='primal',
                          min_search_proportion=1, threshold_proportion=None)
        self.assertRaises(AssertionError, mb._instantiate_model,
                          solve_type='iterative', warm_start=True, method='auto',
                          min_search_proportion=1, threshold_proportion=None,
                          act_tol=2.1)

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
        mb = MinBisect(30, .5, .1, number_of_cuts=100)
        mb.solve_once('dual')
        self.assertTrue([(0, 'once', 'dual', 'cold', 1, None, None)] == list(mb.data.summary_stats.keys()))
        mb.solve_iteratively(min_search_proportion=.1, act_tol=.1)
        self.assertTrue([(0, 'once', 'dual', 'cold', 1, None, None),
                         (0, 'iterative', 'dual', 'warm', .1, None, .1)]
                        == list(mb.data.summary_stats.keys()))
        max_constraints = mb.mdl.NumConstrs
        max_variables = mb.mdl.NumVars

        # make a few runs so we can test if gurobi run time calcs are correct
        mb.solve_iteratively(warm_start=False)
        mb.solve_once(method='auto')
        mb.solve_iteratively(method='auto')
        mb.solve_iteratively(warm_start=False, method='auto')
        self.assertTrue(solution_schema.good_tic_dat_object(mb.data))

        data = mb.data.summary_stats[0, 'iterative', 'dual', 'warm', .1, None, .1]
        self.assertTrue(data['n'] == 30)
        self.assertTrue(data['p'] == .5)
        self.assertTrue(data['q'] == .1)
        self.assertTrue(data['cut_type'] == 'fixed')
        self.assertTrue(data['cut_value'] == 100)
        self.assertTrue(data['max_constraints'] == max_constraints)
        self.assertTrue(data['max_variables'] == max_variables)
        self.assertTrue(data['total_cpu_time'] >= data['gurobi_cpu_time'])
        self.assertTrue(data['total_cpu_time'] >= data['non_gurobi_cpu_time'])
        gurobi_cpu_time = sum(
            d['cpu_time'] for (si, st, m, ws, msp, tp, at, ssi), d in
            mb.data.run_stats.items() if st == 'iterative' and m == 'dual' and
            ws == 'warm' and msp == .1 and tp is None and at == .1
        )
        self.assertTrue(data['gurobi_cpu_time'] == gurobi_cpu_time)
        self.assertTrue(data['total_cpu_time'] == data['gurobi_cpu_time'] + data['non_gurobi_cpu_time'])
        # compares different runs but they should all be same anyways
        self.assertTrue(isclose(data['objective_value'], mb.mdl.ObjVal, abs_tol=.0001))

        # make sure all time values > 0
        for f in mb.data.summary_stats.values():
            self.assertTrue(f['gurobi_cpu_time'] >= 0)
            self.assertTrue(f['non_gurobi_cpu_time'] >= 0)

        # make sure all objective values are the same for each run
        objs = [f['objective_value'] for f in mb.data.summary_stats.values()]
        self.assertTrue(all(isclose(obj, objs[0], abs_tol=.0001) for obj in objs))

    def test_optimize(self):
        mb = MinBisect(8, .5, .1, .1, write_mps=True)
        mb._instantiate_model()
        mb._optimize()
        self.assertTrue([(0, 'iterative', 'dual', 'warm', 1, None, None, 0)] ==
                        list(mb.data.run_stats.keys()))

        # tests adds a second correctly
        ((i, j, k), t) = list(mb.c)[0]
        mb.inf = [((i, j, k), t)]
        mb._add_triangle_inequality(i, j, k, t)
        mb._optimize()
        self.assertTrue([(0, 'iterative', 'dual', 'warm', 1, None, None, 0),
                         (0, 'iterative', 'dual', 'warm', 1, None, None, 1)]
                        == list(mb.data.run_stats.keys()))

        # tests adds all at once solve correctly
        mb.solve_once(method='dual')
        self.assertTrue([(0, 'iterative', 'dual', 'warm', 1, None, None, 0),
                         (0, 'iterative', 'dual', 'warm', 1, None, None, 1),
                         (0, 'once', 'dual', 'cold', 1, None, None, 0)] ==
                        list(mb.data.run_stats.keys()))
        self.assertTrue(solution_schema.good_tic_dat_object(mb.data))

        # check data filled out as expected
        data = mb.data.run_stats[0, 'iterative', 'dual', 'warm', 1, None, None, 1]
        self.assertTrue(data['n'] == 8)
        self.assertTrue(data['p'] == .5)
        self.assertTrue(data['q'] == .1)
        self.assertTrue(data['cut_type'] == 'proportion')
        self.assertTrue(data['cut_value'] == .1)
        self.assertTrue(data['cuts_sought'] == int(.1*224))  # 224 cuts for problem size
        self.assertTrue(data['cuts_added'] == 1)  # only added one manually
        self.assertTrue(data['cuts_removed'] == 0)  # never called remove constraints
        self.assertTrue(data['constraints'] == 2)  # because equal partition and 1 cut
        self.assertTrue(data['variables'] == mb.mdl.NumVars)
        self.assertTrue(data['cpu_time'] >= 0)
        self.assertTrue(data['search_proportion_used'] == 1)
        self.assertTrue(data['current_threshold'] is None)

        # check mps files created
        for pth in ['model_iterative_dual_warm_1_None_0.mps',
                    'model_iterative_dual_warm_1_None_1.mps',
                    'model_once_dual_0.mps',
                    'model_iterative_dual_warm_1_None_1.bas']:
            self.assertTrue(os.path.exists(pth))
            os.remove(pth)

    def test_optimize_captures_correct_cuts(self):
        mb = MinBisect(20, .5, .1, number_of_cuts=10, first_iteration_cuts=500)
        mb.solve_iteratively(act_tol=.1)

        # first iteration should match the fixed number provided
        data = mb.data.run_stats[0, 'iterative', 'dual', 'warm', 1, None, .1, 0]
        self.assertTrue(data['cuts_added'] == 500)
        self.assertTrue(data['cuts_sought'] == 500)
        self.assertTrue(data['constraints'] == 501)
        # and have no cuts removed
        self.assertTrue(data['cuts_removed'] == 0)
        pc = data['constraints']

        # (second and) third iteration should match number of cuts
        data = mb.data.run_stats[0, 'iterative', 'dual', 'warm', 1, None, .1, 1]
        self.assertTrue(data['cuts_added'] == 10)
        self.assertTrue(data['cuts_sought'] == 10)
        self.assertTrue(data['cuts_removed'] > 0)
        # and total cuts should equal last total + added - removed
        self.assertTrue(data['constraints'] ==
                        pc + data['cuts_added'] - data['cuts_removed'])

    def test_optimize_cut_math_right(self):
        mb = MinBisect(8, .5, .1, number_of_cuts=10)
        mb.first_iteration_cuts = 20
        mb.solve_once(method='dual')
        mb.solve_iteratively()
        d = mb.data.run_stats
        self.assertTrue(d[0, 'iterative', 'dual', 'warm', 1, None, None, 0]['cuts_sought'] ==
                        d[0, 'iterative', 'dual', 'warm', 1, None, None, 0]['cuts_added'],
                        'cuts sought and added should be same on first iteration')
        self.assertTrue(d[0, 'iterative', 'dual', 'warm', 1, None, None, 0]['cuts_sought'] == 20,
                        'cuts first sought should be more than other iterations')
        self.assertTrue(d[0, 'once', 'dual', 'cold', 1, None, None, 0]['cuts_sought'] ==
                        d[0, 'once', 'dual', 'cold', 1, None, None, 0]['cuts_added'] == 224)
        self.assertTrue(d[0, 'iterative', 'dual', 'warm', 1, None, None, 1]['cuts_sought'] == 10)

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
        mb = MinBisect(8, .5, .1, number_of_cuts=10)
        mb.solve_once()
        once_obj = mb.mdl.ObjVal
        mb.solve_iteratively()
        self.assertTrue(isclose(once_obj, mb.mdl.ObjVal, abs_tol=.0001),
                        f'one go obj {once_obj} but iterative obj {mb.mdl.ObjVal}')

        mb.solve_iteratively(min_search_proportion=.001)
        self.assertTrue(isclose(once_obj, mb.mdl.ObjVal, abs_tol=.0001),
                        f'one go obj {once_obj} but iterative obj {mb.mdl.ObjVal}')

        mb.solve_iteratively(threshold_proportion=.9)
        self.assertTrue(isclose(once_obj, mb.mdl.ObjVal, abs_tol=.0001),
                        f'one go obj {once_obj} but iterative obj {mb.mdl.ObjVal}')

        mb.solve_iteratively(act_tol=.1)
        self.assertTrue(isclose(once_obj, mb.mdl.ObjVal, abs_tol=.0001),
                        f'one go obj {once_obj} but iterative obj {mb.mdl.ObjVal}')

    def test_solve_iteratively_matches_solve_once_big(self):
        mb = MinBisect(40, .5, .1, number_of_cuts=100)
        mb.solve_once()
        once_obj = mb.mdl.ObjVal
        mb.solve_iteratively()
        self.assertTrue(isclose(once_obj, mb.mdl.ObjVal, abs_tol=.0001),
                        f'one go obj {once_obj} but iterative obj {mb.mdl.ObjVal}')

        mb.solve_iteratively(min_search_proportion=.001)
        self.assertTrue(isclose(once_obj, mb.mdl.ObjVal, abs_tol=.0001),
                        f'one go obj {once_obj} but iterative obj {mb.mdl.ObjVal}')

        mb.solve_iteratively(threshold_proportion=.9)
        self.assertTrue(isclose(once_obj, mb.mdl.ObjVal, abs_tol=.0001),
                        f'one go obj {once_obj} but iterative obj {mb.mdl.ObjVal}')

        mb.solve_iteratively(act_tol=.1)
        self.assertTrue(isclose(once_obj, mb.mdl.ObjVal, abs_tol=.0001),
                        f'one go obj {once_obj} but iterative obj {mb.mdl.ObjVal}')

    def test_solve_iteratively_matches_slim_min_bisection(self):

        # test normal
        x, obj_val, a = solve_iterative_min_bisect(n=40, p=.5, q=.1, cut_size=100)
        mb = MinBisect(n=40, p=.5, q=.1, number_of_cuts=100)
        mb.a = a
        mb.solve_iteratively(method='auto')
        self.assertTrue(isclose(obj_val, mb.mdl.ObjVal, abs_tol=.0001),
                        f'slim obj {obj_val} but iterative obj {mb.mdl.ObjVal}')
        for (i, j) in x:
            self.assertTrue(isclose(x[i, j], mb.x[i, j].x, abs_tol=.0001),
                            f'slim x[{i, j}] {x[i, j]} but iterative x[{i, j}]'
                            f'{mb.x[i, j].x}')

        # test threshold proportion
        x, obj_val, a = solve_iterative_min_bisect(n=40, p=.5, q=.1, cut_size=100,
                                                   threshold_proportion=.9)
        mb = MinBisect(n=40, p=.5, q=.1, number_of_cuts=100)
        mb.a = a
        mb.solve_iteratively(method='auto', threshold_proportion=.9)
        self.assertTrue(isclose(obj_val, mb.mdl.ObjVal, abs_tol=.0001),
                        f'slim obj {obj_val} but iterative obj {mb.mdl.ObjVal}')
        for (i, j) in x:
            self.assertTrue(isclose(x[i, j], mb.x[i, j].x, abs_tol=.0001),
                            f'slim x[{i, j}] {x[i, j]} but iterative x[{i, j}]'
                            f'{mb.x[i, j].x}')

    def test_get_cut_depth(self):
        a = np.array([[0, 1, 0, 1],
                      [1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [1, 0, 1, 0]])
        mb = MinBisect(4, .8, .5, .1)
        mb.a = a

        mb._instantiate_model('iterative')
        mb.mdl.optimize()
        # makes sure that the cut calc is right
        for (i, j, k) in {(i, j, k) for ((i, j, k), t) in mb.c}:
            self.assertTrue(mb._get_cut_depth(i, j, k, 1) ==
                            mb.x[j, k].x - mb.x[i, j].x - mb.x[i, k].x)
            self.assertTrue(mb._get_cut_depth(i, j, k, 2) ==
                            mb.x[i, k].x - mb.x[i, j].x - mb.x[j, k].x)
            self.assertTrue(mb._get_cut_depth(i, j, k, 3) ==
                            mb.x[i, j].x - mb.x[i, k].x - mb.x[j, k].x)
            self.assertTrue(mb._get_cut_depth(i, j, k, 4) ==
                            mb.x[i, j].x + mb.x[i, k].x + mb.x[j, k].x - 2)

    def test_recalibrate_cut_depths_by_threshold_proportion(self):
        a = np.array([[0, 1, 0, 1, 0, 0, 0, 0],
                      [1, 0, 1, 0, 0, 0, 0, 1],
                      [0, 1, 0, 0, 0, 1, 1, 0],
                      [1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 1],
                      [0, 0, 1, 0, 0, 0, 0, 1],
                      [0, 0, 1, 0, 1, 0, 0, 1],
                      [0, 1, 0, 0, 1, 1, 1, 0]])

        # test we get cut_size cuts when threshold low enough
        mb = MinBisect(6, .8, .1, number_of_cuts=10)
        mb.a = a
        mb._instantiate_model('iterative')
        mb._optimize()
        mb.current_threshold = .9
        mb._recalibrate_cut_depths_by_threshold_proportion()
        self.assertTrue(len(mb.d) == 10, 'stop at 10 when more')
        for ((i, j, k), t), cut_depth in mb.d.items():
            self.assertTrue(cut_depth > .9)
            mb._add_triangle_inequality(i, j, k, t)
        mb._optimize()
        mb.d = {}
        mb.v = {}
        mb._recalibrate_cut_depths_by_threshold_proportion()
        self.assertTrue(len(mb.d) == 2, 'grab what is there when less than 10')

    def test_recalibrate_cut_depths_by_search_proportion(self):
        a = np.array([[0, 1, 0, 1],
                      [1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [1, 0, 1, 0]])

        # test full proportion
        mb = MinBisect(4, .8, .5, .1)
        mb.a = a

        mb._instantiate_model('iterative')
        mb.mdl.optimize()
        inf_cuts = {((1, 2, 3), 4), ((0, 1, 3), 1)}
        mb._recalibrate_cut_depths_by_search_proportion()
        self.assertFalse(inf_cuts.difference(mb.d.keys()))
        self.assertFalse(set(mb.d.keys()).difference(inf_cuts))
        self.assertTrue(len(mb.d) == 2, 'only two constraints should be violated')

        # test search proportion < 1
        mb = MinBisect(4, .8, .5, .1)
        mb.a = a

        mb._instantiate_model('iterative')
        mb.mdl.optimize()
        mb.current_search_proportion = .1
        with patch.object(mb, '_get_cut_depth', return_value=1) as cut_depth_mock:
            mb._recalibrate_cut_depths_by_search_proportion()
            self.assertTrue(cut_depth_mock.call_count == 1,
                            'we should only get one iteration when searching 10%')

    def test_recalibrate_cut_depths_by_search_proportion_uses_tolerance(self):
        mb = MinBisect(4, .5, .1, number_of_cuts=20)
        mb._instantiate_model('iterative')
        mb.mdl.update()
        for (i, j), var in mb.x.items():
            var.lb = -1
            mb.mdl.addConstr(mb.x[i, j] == -.00001)
        mb.mdl.remove(mb.mdl.getConstrByName('equal_partitions'))
        mb.mdl.optimize()

        mb._recalibrate_cut_depths_by_search_proportion()
        mb._find_most_violated_constraints()
        self.assertTrue(len(mb.inf) == 0,
                        'all cut depths should be < tolerance')

        mb.tolerance = .000001
        mb._recalibrate_cut_depths_by_search_proportion()
        mb._find_most_violated_constraints()
        self.assertTrue(len(mb.inf) == 12,
                        'only last triangle inequalities should be violated')

    def test_find_new_threshold(self):
        mb = MinBisect(40, .5, .1, number_of_cuts=1000)
        mb._instantiate_model(threshold_proportion=.9)

        for ((i, j, k), t) in random.sample(mb.c, 2000):
            mb._add_triangle_inequality(i, j, k, t)
        mb._optimize()

        mb.d = {}
        mb.v = {}
        mb._recalibrate_cut_depths_by_search_proportion()
        mb._find_new_threshold()
        p = sum(1 for k, v in mb.d.items() if v < mb.current_threshold)/len(mb.d)
        self.assertTrue(isclose(p, mb.threshold_proportion, abs_tol=.01))

    def test_find_most_violated_constraints(self):
        # use 100 to ensure we get some values less than 1 to test that sort is right
        mb = MinBisect(52, .5, .1, number_of_cuts=100)
        mb._instantiate_model('iterative')
        mb.first_iteration_cuts = 5000
        mb.inf = random.sample(mb.c, min(mb.first_iteration_cuts, len(mb.c)))
        for ((i, j, k), t) in mb.inf:
            mb._add_triangle_inequality(i, j, k, t)
        mb._optimize()
        mb._find_most_violated_constraints()
        self.assertTrue(len(mb.inf) == 100, 'we only add 100 cuts at once')
        for k in random.sample(mb.inf, 20):
            self.assertTrue(len([_ for _ in mb.d if mb.d[_] > mb.d[k]]) < 100,
                            'there should be fewer than 100 constraints more violated')

        # make sure it just returns d.keys() if number of cuts is huge
        mb.cut_size = 10000000
        mb._find_most_violated_constraints()
        self.assertTrue(set(mb.inf) == set(mb.d.keys()))

    def test_find_most_violated_constraints_logic_correct(self):
        mb = MinBisect(20, .5, .1, number_of_cuts=10)

        # search proportions iterates
        mb._instantiate_model(min_search_proportion=.1)
        with patch.object(mb, '_recalibrate_cut_depths_by_search_proportion') as ps,\
                patch.object(mb, '_recalibrate_cut_depths_by_threshold_proportion') as pt,\
                patch.object(mb, '_find_new_threshold') as pf:
            mb._find_most_violated_constraints()
            self.assertTrue(pt.call_count == 0, 'pt should not called w/o current threshold')
            self.assertTrue(ps.call_count == 2, 'ps called for each search prop')
            self.assertTrue(pf.call_count == 0, 'ps called for each search prop')
            self.assertFalse(mb.keep_iterating, 'len(inf) == 0 => stop iterating')

        # current threshold
        mb._instantiate_model(threshold_proportion=.9)
        mb.current_threshold = 1
        with patch.object(mb, '_recalibrate_cut_depths_by_search_proportion') as ps, \
                patch.object(mb, '_recalibrate_cut_depths_by_threshold_proportion') as pt, \
                patch.object(mb, '_find_new_threshold') as pf:
            mb._find_most_violated_constraints()
            self.assertTrue(pt.call_count == 1, 'pt called once with current threshold')
            self.assertTrue(ps.call_count == 0, 'ps not called with current threshold')
            self.assertTrue(pf.call_count == 0, 'ps not called with current threshold')
            self.assertTrue(mb.keep_iterating)

        # current threshold that can find cut_size more violated
        mb._instantiate_model(threshold_proportion=.9)
        mb.current_threshold = 1
        mb.mdl.optimize()
        mb._find_most_violated_constraints()
        self.assertTrue(len(mb.inf) == 10)
        self.assertTrue(mb.current_threshold == 1,
                        'current threshold shouldnt change unless len(mb.inf) < 10')
        self.assertTrue(mb.keep_iterating)

        # current threshold that cannot find cut_size more violated
        mb._instantiate_model(threshold_proportion=.9)
        mb.current_threshold = 1.1
        mb.mdl.optimize()
        mb._find_most_violated_constraints()
        self.assertFalse(mb.inf, 'no infeasible with depth >= 1.1')
        self.assertFalse(mb.current_threshold, 'current thresh goes away with no mb.inf')
        self.assertTrue(mb.keep_iterating)

        # no current threshold but thresh prop
        mb._instantiate_model(threshold_proportion=.9)
        mb.mdl.optimize()
        with patch.object(mb, '_recalibrate_cut_depths_by_search_proportion') as ps, \
                patch.object(mb, '_recalibrate_cut_depths_by_threshold_proportion') as pt, \
                patch.object(mb, '_find_new_threshold') as pf:
            mb._find_most_violated_constraints()
            self.assertTrue(pt.call_count == 0, 'pt not called w/o current threshold')
            self.assertTrue(ps.call_count == 1, 'ps only called once w/ thresh prop')
            self.assertTrue(pf.call_count == 0, 'pf not called w/o current threshold')

        # no current threshold but thresh prop finds new threshold
        mb._instantiate_model(threshold_proportion=.9)
        mb.mdl.optimize()
        mb._find_most_violated_constraints()
        self.assertTrue(mb.current_threshold == 1)
        self.assertTrue(mb.keep_iterating)

        # no current threshold but thresh prop cannot find new threshold
        mb = MinBisect(20, .5, .1, number_of_cuts=1000000)
        mb._instantiate_model(threshold_proportion=.9)
        mb.mdl.optimize()
        mb._find_most_violated_constraints()
        self.assertTrue(mb.current_threshold is None)
        self.assertTrue(mb.keep_iterating)

        # search proportions stops early - instantiate one that will give a lot of 1's
        mb = MinBisect(20, .5, .1, number_of_cuts=10)
        mb._instantiate_model(min_search_proportion=.1)
        mb.mdl.optimize()
        mb._find_most_violated_constraints()
        self.assertTrue(mb.current_threshold is None)
        self.assertTrue(mb.keep_iterating)
        self.assertTrue(mb.current_search_proportion == .1)
        self.assertTrue(len(mb.inf) == 10)

    def test_find_most_violated_constraints_when_it_selects_all(self):
        mb = MinBisect(20, .5, .1, number_of_cuts=5000)
        mb._instantiate_model()
        mb._optimize()
        mb._find_most_violated_constraints()
        # when using only cardinality constraint, 1 is only positive cut value
        self.assertTrue(len(mb.inf) == len([k for k, v in mb.d.items() if v == 1]),
                        'select all infeasible cuts when total less than sought')

    def test_remove_inactive_constraints(self):
        mb = MinBisect(20, .5, .1, number_of_cuts=100)
        mb._instantiate_model(act_tol=.1)
        for ((i, j, k), t) in random.sample(mb.c, 100):
            mb._add_triangle_inequality(i, j, k, t)
        mb.mdl.optimize()
        mb._find_most_violated_constraints()
        for ((i, j, k), t) in mb.inf:
            mb._add_triangle_inequality(i, j, k, t)
        mb._optimize()
        mb._remove_inactive_constraints()
        mb.d, mb.v = {}, {}
        self.assertTrue(mb.removed)
        for ((i, j, k), t) in create_constraint_indices(range(20)).difference(mb.c):
            if ((i, j, k), t) in mb.removed:
                # may need to adjust tolerances here
                self.assertTrue(mb._get_cut_depth(i, j, k, t) < -mb.act_tol,
                                'only remove constraints not close to being active')
            else:
                self.assertTrue(mb.mdl.getConstrByName(f'{i}_{j}_{k}_tri{t}'),
                                'if not in c, it should be in the model since not removed')
                self.assertTrue(1e-10 >= mb._get_cut_depth(i, j, k, t) >= -mb.act_tol,
                                'constraints close to active should be left')

    def test_iterate(self):
        mb = MinBisect(20, .5, .1, number_of_cuts=100)
        mb._instantiate_model(act_tol=.1)
        for ((i, j, k), t) in random.sample(mb.c, 100):
            mb._add_triangle_inequality(i, j, k, t)
        mb.mdl.optimize()
        # don't account for mb.ati removing elements since it's patched
        expected_c = {k for k in mb.c}

        # normal iteration
        mb._remove_inactive_constraints()
        self.assertTrue(mb.removed)  # need to have some constraints removed for test to work
        expected_c.update(mb.removed)
        mb._find_most_violated_constraints()
        self.assertTrue(mb.inf)  # need to have some upcoming infeasible const for test to work
        with patch.object(mb, '_find_most_violated_constraints') as fmvc, \
                patch.object(mb, '_add_triangle_inequality') as ati, \
                patch('min_bisection.gu.Model.reset') as r, \
                patch.object(mb, '_optimize') as o, \
                patch.object(mb, '_remove_inactive_constraints') as ric:
            mb._iterate()
            self.assertTrue(fmvc.call_count == 1)
            self.assertTrue(expected_c == mb.c)
            self.assertTrue(ati.call_count == len(mb.inf))
            self.assertTrue(r.call_count == 0)
            self.assertTrue(o.call_count == 1)
            self.assertTrue(ric.call_count == 1)

    def test_iterate_terminates(self):
        mb = MinBisect(20, .5, .1, number_of_cuts=100)
        mb._instantiate_model(act_tol=.1)
        for ((i, j, k), t) in random.sample(mb.c, 100):
            mb._add_triangle_inequality(i, j, k, t)
        mb.mdl.optimize()
        mb._remove_inactive_constraints()
        self.assertTrue(mb.removed)  # need to have some constraints removed for test to work
        mb._find_most_violated_constraints()
        self.assertTrue(mb.inf)  # need to have some upcoming infeasible const for test to work
        prev_c = {k for k in mb.c}  # account for mb.ati removing elements

        # final iteration should end early
        with patch.object(mb, '_find_most_violated_constraints') as fmvc, \
                patch.object(mb, '_add_triangle_inequality') as ati, \
                patch('min_bisection.gu.Model.reset') as r, \
                patch.object(mb, '_optimize') as o, \
                patch.object(mb, '_remove_inactive_constraints') as ric:
            mb.keep_iterating = False
            mb._iterate()
            self.assertTrue(fmvc.call_count == 1)
            self.assertTrue(prev_c == mb.c, 'no cuts should have been removed or added')
            self.assertTrue(ati.call_count == 0)
            self.assertTrue(r.call_count == 0)
            self.assertTrue(o.call_count == 0)
            self.assertTrue(ric.call_count == 0)

    def test_iterate_skips_remove_constraints(self):
        mb = MinBisect(20, .5, .1, number_of_cuts=100)
        mb._instantiate_model()
        for ((i, j, k), t) in random.sample(mb.c, 100):
            mb._add_triangle_inequality(i, j, k, t)
        mb.mdl.optimize()

        # final iteration should end early
        with patch.object(mb, '_find_most_violated_constraints') as fmvc, \
                patch.object(mb, '_optimize') as o, \
                patch.object(mb, '_remove_inactive_constraints') as ric:
            mb._iterate()
            self.assertTrue(fmvc.call_count == 1)
            self.assertTrue(o.call_count == 1)
            self.assertTrue(ric.call_count == 0)

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
