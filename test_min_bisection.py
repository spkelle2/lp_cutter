import inspect
import numpy as np
import os
import shutil
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
    def test_all_exist(self):
        # make sure all attributes exist and two functions called
        pass


class TestAddTriangleInequality(unittest.TestCase):

    def test_adds_and_deletes(self):
        mb = MinBisect(8, .5, .1, .1)
        mb.instantiate_model()
        ((i, j, k), t) = [k for k in mb.c.keys()][0]
        mb.add_triangle_inequality(i, j, k, t)
        self.assertTrue(mb.mdl.getConstrByName(f'{i}_{j}_{k}_tri{t}'),
                        f'constraint {i}_{j}_{k}_tri{t} should be added')
        self.assertFalse(((i, j, k), t) in mb.c,
                         f"(({i}, {j}, {k}), {t}) should be removed from c")


class TestInstantiateModel(unittest.TestCase):
    def everything_that_should_be_is(self):
        indices = range(8)
        mb = MinBisect(8, .5, .1, .1)
        mb.instantiate_model()
        for i in indices:
            for j in indices:
                if i != j:
                    self.assertTrue("x[i,j] exists")
        self.assertTrue(mb.mdl.getConstrByName(f'Equal Partitions'),
                        f'Equal Partition Constraint should exist')
        # check that objective was set


class TestSolveOnce:
    def is_correct(self):
        # solves to right solution


class TestSolveIteratively:
    def matches_solve_once(self):
        # solves to same solution as solve_once