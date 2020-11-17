import inspect
import numpy as np
import os
import shutil
import unittest

from graph import Graph, GraphException
from min_bisection import input_schema, data_integrity_checks


class TestInit(unittest.TestCase):

    def test_correct_density_within_cluster(self):
        n, p = 100, .5
        graph = Graph(n, p, .1)

        # ratio of actual edge count in cluster1 to max possible edge count in cluster1
        p1 = graph.a[:n//2, :n//2].sum()/((n//2)**2 - n//2)
        self.assertTrue(.45 <= p1 <= .55,
                        msg=(f'a large intracluster density should be near p={p}'))

        # ratio of actual edge count in cluster2 to max possible edge count in cluster2
        p2 = graph.a[n//2:, n//2:].sum()/((n//2)**2 - n//2)
        self.assertTrue(.45 <= p2 <= .55,
                        msg=(f'a large intracluster density should be near p={p}'))

    def test_correct_density_between_clusters(self):
        n, q = 100, .1
        graph = Graph(n, .5, q)

        # ratio of actual edge count between clusters to max possible edge count between clusters
        k = graph.a[:n//2, n//2:].sum()/((n//2)**2 - n//2)
        self.assertTrue(.05 <= k <= .15,
                        msg=(f'large a intercluster density should be near q={q}'))

    def test_reasonable_values(self):
        graph = Graph(10, .5, .1)
        self.assertTrue(set(np.unique(graph.a)) == {0, 1}, 'only values are 0 and 1')
        self.assertTrue(graph.a.trace() == 0, 'no values should exist on diagonal')
        self.assertTrue((graph.a == graph.a.T).all(), 'transpose should be equal')

    def test_correct_dimension(self):
        graph = Graph(10, .5, .1)
        self.assertTrue(graph.a.shape == (10, 10), 'the dimension should be n by n')

    def test_cluster_sizes_correct(self):
        # ensure cluster sizes are correct by forcing all edges within a cluster
        # and none between then counting that the total number in each quadrant correct
        graph = Graph(10, 1, 0)
        self.assertTrue(graph.a[:5, :5].sum() == 20)
        self.assertTrue(graph.a[5:, 5:].sum() == 20)
        self.assertTrue(graph.a[:5, 5:].sum() == 0)
        self.assertTrue(graph.a[5:, :5].sum() == 0)

        graph = Graph(11, 1, 0)
        self.assertTrue(graph.a[:5, :5].sum() == 20)
        self.assertTrue(graph.a[5:, 5:].sum() == 30)
        self.assertTrue(graph.a[:5, 5:].sum() == 0)
        self.assertTrue(graph.a[5:, :5].sum() == 0)

    def test_bad_input(self):
        # n <= 0 should fail
        self.assertRaises(GraphException, Graph, n=-5, p=.5, q=.1)
        # n not integer should fail
        self.assertRaises(GraphException, Graph, n=2.3, p=.5, q=.1)
        # p not in [0, 1] should fail
        self.assertRaises(GraphException, Graph, n=10, p=1.1, q=.1)
        # q not in [0, 1] should fail
        self.assertRaises(GraphException, Graph, n=-5, p=.5, q=1.1)
        # p not a number should fail
        self.assertRaises(GraphException, Graph, n=-5, p='one-half', q=.1)
        # q not a number should fail
        self.assertRaises(GraphException, Graph, n=-5, p=.5, q='one-half')


class TestSave(unittest.TestCase):

    fldr_pth = os.path.join(os.path.dirname(inspect.getfile(TestInit)), 'data')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.fldr_pth, ignore_errors=True)

    def setUp(self):
        self.tearDownClass()
        os.mkdir(self.fldr_pth)

    def tearDown(self):
        self.tearDownClass()

    def test_bad_save_inputs(self):
        graph = Graph(4, .5, .1)
        # cut_prop not number should fail
        self.assertRaises(GraphException, graph.save, fldr=self.fldr_pth,
                          cut_proportion='one-half')
        # cut_prop improper number should fail
        self.assertRaises(GraphException, graph.save, fldr=self.fldr_pth,
                          cut_proportion=-1)
        # non existent folder path should fail
        self.assertRaises(GraphException, graph.save, fldr='bad_pth')

    def test_good_save(self):
        """If a graph saves properly, it should pass input schema's integrity tests"""
        graph = Graph(4, .5, .1)
        graph.save(self.fldr_pth)
        dat = input_schema.csv.create_tic_dat(self.fldr_pth)
        data_integrity_checks(dat)


if __name__ == '__main__':
    unittest.main()
