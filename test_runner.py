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
    min_orders = [0, 1]
    fldr = 'test_output'

    def tearDown(self):
        shutil.rmtree(self.fldr, ignore_errors=True)

    def test_saves_exactly_all_run_data(self):
        run_experiments(self.ns, self.ps, self.qs, self.cut_proportions,
                        self.numbers_of_cuts, self.min_orders, self.repeats,
                        self.fldr)
        data = solution_schema.csv.create_tic_dat(self.fldr)

        # the three valid combos should each run once and iteratively
        # min_orders = 1 and cut_proportions = .1 is an invalid combo
        self.assertTrue(len(data.summary_stats) == 6)
        for (cut_value, min_order) in \
                product(self.cut_proportions + self.numbers_of_cuts, self.min_orders):
            matches = {pk: f for pk, f in data.summary_stats.items() if
                       f['cut_value'] == cut_value and f['min_order'] == min_order}
            if cut_value == .1 and min_order == 1:
                self.assertFalse(matches)
            else:
                self.assertTrue(len(matches) == 2)


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
