import numpy as np
import unittest

from graph import two_clustered_graph, GraphException


class TestTwoClusteredGraph(unittest.TestCase):

    def test_correct_density_within_cluster(self):
        n, p = 100, .5
        a = two_clustered_graph(n, p, .1)

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
        a = two_clustered_graph(n, .5, q)

        # ratio of actual edge count between clusters to max possible edge count between clusters
        k = a[:n//2, n//2:].sum()/((n//2)**2 - n//2)
        self.assertTrue(.05 <= k <= .15,
                        msg=(f'large a intercluster density should be near q={q}'))

    def test_reasonable_values(self):
        a = two_clustered_graph(10, .5, .1)
        self.assertTrue(set(np.unique(a)) == {0, 1}, msg='only values are 0 and 1')
        self.assertTrue(a.trace() == 0, 'no values should exist on diagonal')
        self.assertTrue((a == a.T).all(), 'transpose should be equal')

    def test_correct_dimension(self):
        a = two_clustered_graph(10, .5, .1)
        self.assertTrue(a.shape == (10, 10), msg=('the dimension should be n by n'))

    def test_cluster_sizes_correct(self):
        # ensure cluster sizes are correct by forcing all edges within a cluster
        # and none between then counting that the total number in each quadrant correct
        a = two_clustered_graph(10, 1, 0)
        self.assertTrue(a[:5, :5].sum() == 20)
        self.assertTrue(a[5:, 5:].sum() == 20)
        self.assertTrue(a[:5, 5:].sum() == 0)
        self.assertTrue(a[5:, :5].sum() == 0)

        a = two_clustered_graph(11, 1, 0)
        self.assertTrue(a[:5, :5].sum() == 20)
        self.assertTrue(a[5:, 5:].sum() == 30)
        self.assertTrue(a[:5, 5:].sum() == 0)
        self.assertTrue(a[5:, :5].sum() == 0)

    def test_bad_input(self):
        # n <= 0 should fail
        self.assertRaises(GraphException, two_clustered_graph, n=-5, p=.5, q=.1)
        # n not integer should fail
        self.assertRaises(GraphException, two_clustered_graph, n=2.3, p=.5, q=.1)
        # p not in [0, 1] should fail
        self.assertRaises(GraphException, two_clustered_graph, n=10, p=1.1, q=.1)
        # q not in [0, 1] should fail
        self.assertRaises(GraphException, two_clustered_graph, n=-5, p=.5, q=1.1)
        # p not a number should fail
        self.assertRaises(GraphException, two_clustered_graph, n=-5, p='one-half', q=.1)
        # q not a number should fail
        self.assertRaises(GraphException, two_clustered_graph, n=-5, p=.5, q='one-half')


if __name__ == '__main__':
    unittest.main()
