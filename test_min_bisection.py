from math import isclose
import numpy as np
import unittest

from min_bisection import create_constraint_indices, create_adjacency_matrix, MinBisect


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


class TestInit(unittest.TestCase):
    def test_proper_cut_size(self):
        mb = MinBisect(8, .5, .1, cut_proportion=.1)
        self.assertTrue(mb.cut_size == int(.1*len(mb.c)))
        mb = MinBisect(8, .5, .1, number_of_cuts=10)
        self.assertTrue(mb.cut_size == 10)


class TestAddTriangleInequality(unittest.TestCase):
    def test_adds_and_deletes(self):
        mb = MinBisect(8, .5, .1, .1)
        mb.instantiate_model()
        ((i, j, k), t) = [k for k in mb.c.keys()][0]
        mb.add_triangle_inequality(i, j, k, t)
        mb.mdl.update()
        self.assertTrue(mb.mdl.getConstrByName(f'{i}_{j}_{k}_tri{t}'),
                        f'constraint {i}_{j}_{k}_tri{t} should be added')
        self.assertFalse(((i, j, k), t) in mb.c,
                         f"(({i}, {j}, {k}), {t}) should be removed from c")


class TestInstantiateModel(unittest.TestCase):
    def test_everything_that_should_be_is(self):
        indices = range(8)
        mb = MinBisect(8, .5, .1, .1)
        mb.instantiate_model()
        for i in indices:
            for j in indices:
                if i != j:
                    self.assertTrue((i, j) in mb.x,
                                    'any i != j should be in x')
        mb.mdl.update()
        self.assertTrue(mb.mdl.getConstrByName(f'Equal Partitions'),
                        'Equal Partition Constraint should exist')
        self.assertTrue(mb.mdl.getObjective(),
                        'Objective should be set')


class TestSolveOnce(unittest.TestCase):
    def test_is_correct(self):
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
        mb.instantiate_model()
        mb.solve_once()
        self.assertTrue(mb.mdl.ObjVal == 3, 'only three edges cross clusters')
        self.assertTrue(mb.x[0, 1].x == mb.x[0, 2].x == mb.x[0, 3].x == 0,
                        '0, 1, 2, and 3 should be one cluster')
        self.assertTrue(mb.x[4, 5].x == mb.x[4, 6].x == mb.x[4, 7].x == 0,
                        '4, 5, 6, and 7 should be one cluster')


class TestSolveIteratively(unittest.TestCase):
    def test_matches_solve_once_small(self):
        mbo = MinBisect(8, .5, .1, .1)
        mbi = MinBisect(8, .5, .1, .1)
        mbi.a = mbo.a
        mbo.instantiate_model()
        mbi.instantiate_model()
        mbo.solve_once()
        mbi.solve_iteratively()
        self.assertTrue(isclose(mbo.mdl.ObjVal, mbi.mdl.ObjVal, rel_tol=1e-3),
                        f'one go obj {mbo.mdl.ObjVal} but iterative obj {mbi.mdl.ObjVal}')

    def test_matches_solve_once_big(self):
        mbo = MinBisect(40, .5, .1, .1)
        mbi = MinBisect(40, .5, .1, .1)
        mbi.a = mbo.a
        mbo.instantiate_model()
        mbi.instantiate_model()
        mbo.solve_once()
        mbi.solve_iteratively()
        self.assertTrue(isclose(mbo.mdl.ObjVal, mbi.mdl.ObjVal, rel_tol=1e-3),
                        f'one go obj {mbo.mdl.ObjVal} but iterative obj {mbi.mdl.ObjVal}')


if __name__ == '__main__':
    unittest.main()