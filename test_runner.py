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
    warm_starts = [True, False]
    methods = ['dual', 'auto']
    fldr = 'test_output'

    def tearDown(self):
        shutil.rmtree(self.fldr, ignore_errors=True)

    def test_saves_exactly_all_run_data(self):
        run_experiments(self.ns, self.ps, self.qs, self.cut_proportions,
                        self.numbers_of_cuts, self.repeats, self.warm_starts,
                        self.methods, self.fldr)
        data = solution_schema.csv.create_tic_dat(self.fldr)

        # for each parameter combo we made we should see a certain number of runs recorded
        for n, p, q, cut_value in product(self.ns, self.ps, self.qs,
                                          self.cut_proportions + self.numbers_of_cuts):
            matching_rows = {pk: f for pk, f in data.summary_stats.items() if
                             f['n'] == n and f['p'] == p and f['q'] == q and
                             f['cut_value'] == cut_value}
            # matching row for each combination of repeat, method, and warm start
            self.assertTrue(len(matching_rows) ==
                            len(self.methods)*(1+len(self.warm_starts))*self.repeats)

        # every combo recorded should belong to a combo we made
        wd = {True: 'warm', False: 'cold'}
        for pk, f in data.summary_stats.items():
            self.assertTrue(f['n'] in self.ns)
            self.assertTrue(f['p'] in self.ps)
            self.assertTrue(f['q'] in self.qs)
            self.assertTrue(f['cut_value'] in self.cut_proportions or
                            f['cut_value'] in self.numbers_of_cuts)
            self.assertTrue(pk[3] in [wd[ws] for ws in self.warm_starts])
            self.assertTrue(pk[2] in self.methods)

    # both also confirm we work without giving warm starts or methods
    def test_works_with_no_proportions(self):
        run_experiments(self.ns, self.ps, self.qs,
                        numbers_of_cuts=self.numbers_of_cuts,
                        repeats=self.repeats, fldr=self.fldr)

    def test_works_with_no_numbers(self):
        run_experiments(self.ns, self.ps, self.qs,
                        cut_proportions=self.cut_proportions,
                        repeats=self.repeats, fldr=self.fldr)


if __name__ == '__main__':
    unittest.main()
