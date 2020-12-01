from itertools import product
import shutil
import unittest

from min_bisection import solution_schema
from runner import run_experiments


class TestRunExperiments(unittest.TestCase):

    ns = [10, 20]
    ps = [.5, .8]
    qs = [.1, .2]
    cut_proportions = [.1]
    numbers_of_cuts = [10]
    repeats = 1
    fldr = 'test_output'

    def tearDown(self):
        shutil.rmtree(self.fldr, ignore_errors=True)

    def test_saves_exactly_all_run_data(self):
        run_experiments(self.ns, self.ps, self.qs, self.cut_proportions,
                        self.numbers_of_cuts, self.repeats, self.fldr)
        data = solution_schema.csv.create_tic_dat(self.fldr)

        for n, p, q, cut_value in product(self.ns, self.ps, self.qs,
                                          self.cut_proportions + self.numbers_of_cuts):
            matching_rows = {pk: f for pk, f in data.summary_stats.items() if
                             f['n'] == n and f['p'] == p and f['q'] == q and
                             f['cut_value'] == cut_value}
            self.assertTrue(len(matching_rows) == 2*self.repeats)

        for f in data.summary_stats.values():
            self.assertTrue(f['n'] in self.ns)
            self.assertTrue(f['p'] in self.ps)
            self.assertTrue(f['q'] in self.qs)
            self.assertTrue(f['cut_value'] in self.cut_proportions or
                            f['cut_value'] in self.numbers_of_cuts)


if __name__ == '__main__':
    unittest.main()
