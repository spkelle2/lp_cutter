import os
import shutil
import unittest

from min_bisection import solution_schema, create_constraint_indices
from runner import run_experiments


class TestRunExperiments(unittest.TestCase):

    ns = [10]
    ps = [.5]
    qs = [.2]
    cut_proportions = [.1]
    numbers_of_cuts = [10]
    remove_constraint_list = [True, False]
    zero_slack_likelihoods = [0, .2]
    repeats = 1
    min_search_proportions = [.1, 1]
    threshold_proportions = [.9]
    fldr = 'test_output'

    def tearDown(self):
        shutil.rmtree(self.fldr, ignore_errors=True)

    def read_data(self):
        data = solution_schema.TicDat()
        for sub_fldr in next(os.walk(self.fldr))[1]:
            current_data = solution_schema.csv.create_tic_dat(os.path.join(self.fldr, sub_fldr))
            for t in solution_schema.all_tables:
                for pk, f in getattr(current_data, t).items():
                    getattr(data, t)[pk] = f
        return data

    def test_saves_exactly_all_run_data(self):
        run_experiments(ns=self.ns, ps=self.ps, qs=self.qs,
                        cut_proportions=self.cut_proportions,
                        numbers_of_cuts=self.numbers_of_cuts,
                        min_search_proportions=self.min_search_proportions,
                        threshold_proportions=self.threshold_proportions,
                        remove_constraint_list=self.remove_constraint_list,
                        zero_slack_likelihoods=self.zero_slack_likelihoods,
                        repeats=self.repeats, fldr=self.fldr)
        data = self.read_data()

        # .1: 1 single and (1 of 3 cut calc)*(3 of 4 remove constraint) = 4
        # 10: 1 single and (3 of 3 cut calc)*(3 of 4 remove constraint) = 10
        self.assertTrue(len(data.summary_stats) == 14)
        self.assertTrue(len([pk for pk, f in data.summary_stats.items() if
                             f['cut_value'] == .1]) == 4)
        self.assertTrue(len([pk for pk, f in data.summary_stats.items() if
                             f['cut_value'] == 10]) == 10)

        # no combos should have memory profile
        # we should have 4 iterative runs with run_time profile
        self.assertTrue(len(os.listdir(os.path.join(self.fldr, 'profiles'))) == 12)
        self.assertTrue(len([x for x in os.listdir(os.path.join(self.fldr, 'profiles'))
                             if x.endswith('_run_time.prof')]) == 12)

    def test_turns_off_single_solve(self):
        run_experiments(ns=self.ns, ps=self.ps, qs=self.qs,
                        cut_proportions=self.cut_proportions,
                        numbers_of_cuts=self.numbers_of_cuts, solve_once=False,
                        min_search_proportions=self.min_search_proportions,
                        threshold_proportions=self.threshold_proportions,
                        remove_constraint_list=self.remove_constraint_list,
                        zero_slack_likelihoods=self.zero_slack_likelihoods,
                        repeats=self.repeats, fldr=self.fldr)
        data = self.read_data()

        # .1: (1 of 3 cut calc)*(3 of 4 remove constraint) = 3
        # 10: (3 of 3 cut calc)*(3 of 4 remove constraint) = 9
        self.assertTrue(len(data.summary_stats) == 12)
        self.assertTrue(len([pk for pk, f in data.summary_stats.items() if
                             f['cut_value'] == .1]) == 3)
        self.assertTrue(len([pk for pk, f in data.summary_stats.items() if
                             f['cut_value'] == 10]) == 9)

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
