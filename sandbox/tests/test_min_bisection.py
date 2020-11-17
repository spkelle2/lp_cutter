from numpy import ndenumerate
import unittest

from graph import Graph
from min_bisection import input_schema, data_integrity_checks


class TestDataIntegrityChecks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.g = Graph(4, .5, .1)

    def test_integrity_failures(self):
        dat = input_schema.TicDat()
        dat.a = {(1, 2): [0, 0]}
        # ensure assertion called on good ticdat
        self.assertRaises(AssertionError, data_integrity_checks, dat)
        dat = input_schema.TicDat()
        dat.a[(1, 2)] = 2
        # ensure assertion called on data type
        self.assertRaises(AssertionError, data_integrity_checks, dat)
        dat.a[(1, 2)] = 1
        # ensure assertion called on missing pair
        self.assertRaises(AssertionError, data_integrity_checks, dat)
        dat.a[(2, 2)] = 1
        # ensure assertion called on data row fail
        self.assertRaises(AssertionError, data_integrity_checks, dat)

    def test_good_dat(self):
        dat = input_schema.TicDat()
        for (i, j), v in ndenumerate(self.g.a):
            if i < j:
                dat.a[(i, j)] = v
        data_integrity_checks(dat)


class TestSolve(unittest.TestCase):
    pass

