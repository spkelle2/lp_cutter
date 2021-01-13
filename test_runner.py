from itertools import product
import shutil
import unittest

from min_bisection import solution_schema, create_constraint_indices
from runner import run_experiments


class TestRunExperiments(unittest.TestCase):

    ns = [10]
    ps = [.5]
    qs = [.1]
    cut_proportions = [.1]
    numbers_of_cuts = [10]
    repeats = 1
    min_search_proportions = [.1, 1]
    threshold_proportions = [.9]
    fldr = 'test_output'

    def tearDown(self):
        shutil.rmtree(self.fldr, ignore_errors=True)

    def test_saves_exactly_all_run_data(self):
        run_experiments(self.ns, self.ps, self.qs, self.cut_proportions,
                        self.numbers_of_cuts, self.min_search_proportions,
                        self.threshold_proportions, self.repeats, self.fldr)
        data = solution_schema.csv.create_tic_dat(self.fldr)

        # 8 possible combos here 2 * (1 + 2 + 1)
        # when cut_proportion = .1, we should not use threshold_proportion = .9
        # or min_search_proportion = .1
        self.assertTrue(len(data.summary_stats) == 6)
        self.assertTrue(len([pk for pk, f in data.summary_stats.items() if
                             f['cut_value'] == .1]) == 2)
        self.assertTrue(len([pk for pk, f in data.summary_stats.items() if
                             f['cut_value'] == 10]) == 4)

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
