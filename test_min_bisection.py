from math import isclose
import numpy as np
import unittest

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
                    if i < j and len({i, j, k}) == len([i, j, k]):
                        self.assertTrue(((i, j, k), 1) in c,
                                        f'i={i}, j={j}, k={k}, t=1 belongs in c')
                        good_idx.add(((i, j, k), 1))
                    if i < j < k:
                        self.assertTrue(((i, j, k), 2) in c,
                                        f'i={i}, j={j}, k={k}, t=1 belongs in c')
                        good_idx.add(((i, j, k), 2))
        self.assertFalse(set(c.keys()).difference(good_idx),
                         'there should not be any more indices')


class TestMinBisection(unittest.TestCase):
    def test_init(self):
        # proportion
        mb = MinBisect(8, .5, .1, cut_proportion=.1)
        self.assertTrue(mb.cut_type == 'proportion')
        self.assertTrue(mb.cut_value == .1)

        # fixed
        mb = MinBisect(8, .5, .1, number_of_cuts=10)
        self.assertTrue(mb.cut_type == 'fixed')
        self.assertTrue(mb.cut_value == 10)

        # bad inputs
        self.assertRaises(AssertionError, MinBisect, 8, .5, .1)
        self.assertRaises(AssertionError, MinBisect, 8, .5, .1, .1, 10)
        self.assertRaises(AssertionError, MinBisect, 7, .5, .1, .1)
        self.assertRaises(AssertionError, MinBisect, 7, 2, .1, .1)
        self.assertRaises(AssertionError, MinBisect, 7, .5, -1, .1)
        self.assertRaises(AssertionError, MinBisect, 7, .5, .1, 10)
        self.assertRaises(AssertionError, MinBisect, 7, .5, .1, None, .1)

    def test_instantiate_model(self):
        indices = range(8)
        mb = MinBisect(8, .5, .1, .1)
        mb._instantiate_model()
        for i in indices:
            for j in indices:
                if i != j:
                    self.assertTrue((i, j) in mb.x, 'any i != j should be in x')
        mb.mdl.update()
        self.assertTrue(mb.mdl.getConstrByName(f'Equal Partitions'),
                        'Equal Partition Constraint should exist')
        self.assertTrue(mb.mdl.getObjective(),
                        'Objective should be set')

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

    def test_triangle_inequality(self):
        mb = MinBisect(8, .5, .1, .1)
        mb._instantiate_model()
        ((i, j, k), t) = [k for k in mb.c.keys()][0]
        mb._add_triangle_inequality(i, j, k, t)
        mb.mdl.update()
        self.assertTrue(mb.mdl.getConstrByName(f'{i}_{j}_{k}_tri{t}'),
                        f'constraint {i}_{j}_{k}_tri{t} should be added')
        self.assertFalse(((i, j, k), t) in mb.c,
                         f"(({i}, {j}, {k}), {t}) should be removed from c")

    def test_summary_profile(self):
        mb = MinBisect(8, .8, .1, .1)
        mb.solve_once()
        self.assertTrue([(0, 'once')] == list(mb.data.summary_stats.keys()))
        mb.solve_iteratively()
        self.assertTrue([(0, 'once'), (0, 'iterative')] == list(mb.data.summary_stats.keys()))
        self.assertTrue(solution_schema.good_tic_dat_object(mb.data))
        data = mb.data.summary_stats[0, 'iterative']
        self.assertTrue(data['n'] == 8)
        self.assertTrue(data['p'] == .8)
        self.assertTrue(data['q'] == .1)
        self.assertTrue(data['cut_type'] == 'proportion')
        self.assertTrue(data['cut_value'] == .1)
        self.assertTrue(data['max_constraints'] == mb.mdl.NumConstrs)
        self.assertTrue(data['max_variables'] == mb.mdl.NumVars)
        self.assertTrue(data['total_cpu_time'] >= data['gurobi_cpu_time'])
        self.assertTrue(data['total_cpu_time'] >= data['non_gurobi_cpu_time'])
        gurobi_cpu_time = sum(d['cpu_time'] for (_, solve_type, _), d in
                              mb.data.run_stats.items() if solve_type == 'iterative')
        self.assertTrue(data['gurobi_cpu_time'] == gurobi_cpu_time)
        self.assertTrue(data['total_cpu_time'] == data['gurobi_cpu_time'] + data['non_gurobi_cpu_time'])
        self.assertTrue(data['objective_value'] == mb.mdl.ObjVal)

    def test_optimize(self):
        mb = MinBisect(8, .5, .1, .1)
        mb.solve_type = 'iterative'
        mb._instantiate_model()
        mb._optimize()
        self.assertTrue([(0, 'iterative', 0)] == list(mb.data.run_stats.keys()))

        # tests adds a second correctly
        ((i, j, k), t) = [k for k in mb.c.keys()][0]
        mb.inf = [((i, j, k), t)]
        mb._add_triangle_inequality(i, j, k, t)
        mb._optimize()
        self.assertTrue([(0, 'iterative', 0), (0, 'iterative', 1)] ==
                        list(mb.data.run_stats.keys()))

        # tests adds all at once solve correctly
        mb.solve_once()
        self.assertTrue([(0, 'iterative', 0), (0, 'iterative', 1), (0, 'once', 0)] ==
                        list(mb.data.run_stats.keys()))
        self.assertTrue(solution_schema.good_tic_dat_object(mb.data))

        # check data filled out as expected
        data = mb.data.run_stats[0, 'iterative', 1]
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

    def test_optimize_cut_math_right(self):
        mb = MinBisect(8, .5, .1, number_of_cuts=10)
        mb.solve_once()
        mb.solve_iteratively()
        d = mb.data.run_stats
        self.assertTrue(d[0, 'iterative', 0]['cuts_sought'] ==
                        d[0, 'iterative', 0]['cuts_added'],
                        'cuts sought and added should be same on first iteration')
        self.assertTrue(d[0, 'iterative', 0]['cuts_sought'] > 10,
                        'cuts first sought should be more than other iterations')
        self.assertTrue(d[0, 'once', 0]['cuts_sought'] ==
                        d[0, 'once', 0]['cuts_added'] == 224)
        self.assertTrue(d[0, 'iterative', 1]['cuts_sought'] == 10)

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

    def test_solve_iteratively_matches_solve_once_big(self):
        mb = MinBisect(40, .5, .1, .1)
        mb.solve_once()
        once_obj = mb.mdl.ObjVal
        mb.solve_iteratively()
        self.assertTrue(isclose(once_obj, mb.mdl.ObjVal, rel_tol=1e-3),
                        f'one go obj {once_obj} but iterative obj {mb.mdl.ObjVal}')

    def test_recalibrate_cut_depths(self):
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


if __name__ == '__main__':
    unittest.main()