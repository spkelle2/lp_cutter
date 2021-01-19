"""
Call with python runner.py <fldr> <n_1> ... <n_m>
"""

from itertools import product
import os
import shutil
import sys

from min_bisection import MinBisect, solution_schema, create_constraint_indices


def run_experiments(ns, ps, qs, cut_proportions=None, numbers_of_cuts=None,
                    min_search_proportions=None, threshold_proportions=None,
                    repeats=1, fldr='run_results', solve_once=True):
    """Solve all at once and solve iteratively the min bisection problem
    <repeats> times for each combo of n, p, q, and cut_proportion/number_of_cuts.

    :param ns: list of problem sizes to test
    :param ps: list of intra-cluster edge likelihoods to try
    :param qs: list of inter-cluster edge likelihoods to try
    :param cut_proportions: list of cut proportions to try for each n, p, q combo
    :param numbers_of_cuts: list of fixed number of cuts to try for each n, p, q combo
    :param min_search_proportions: list of smallest proportion of available cuts
    to search for deepest cuts to add to the model when solving iteratively
    :param threshold_proportions: list of proportions of infeasible cuts one cut
    must be more violated than to be added to the model when solving iteratively
    :param repeats: how many times to rerun each n, p, q, cut_proportion/fixed cuts combo
    :param fldr: where in the working directory to save the results of all
    experiments to csv's
    :param solve_once: True includes single solves in batch runs, False does not.
    :return:
    """
    numbers_of_cuts = [] if not numbers_of_cuts else numbers_of_cuts
    cut_proportions = [] if not cut_proportions else cut_proportions
    min_search_proportions = [1] if not min_search_proportions else min_search_proportions
    threshold_proportions = [] if not threshold_proportions else threshold_proportions

    cuts_possible = {n: len(create_constraint_indices(range(n))) for n in ns}

    shutil.rmtree(fldr, ignore_errors=True)
    os.mkdir(fldr)
    profile_fldr = os.path.join(fldr, 'profiles')
    os.mkdir(profile_fldr)

    combinations = [
        {'n': n, 'p': p, 'q': q, 'cut_proportion': cut_proportion} for
        (n, p, q, cut_proportion) in product(ns, ps, qs, cut_proportions)
    ] + [
        {'n': n, 'p': p, 'q': q, 'number_of_cuts': number_of_cuts} for
        (n, p, q, number_of_cuts) in product(ns, ps, qs, numbers_of_cuts)
    ]
    combinations *= repeats
    combinations.sort(key=lambda x: x['n'])

    for i, combo in enumerate(combinations):
        print(f'running test {i+1} of {len(combinations)}')
        mb = MinBisect(**combo, solve_id=i)
        cut_size = mb.cut_value if mb.cut_type == 'fixed' else \
            int(mb.cut_value * cuts_possible[mb.n])
        base = '_'.join([f'{k}={v}' for k, v in combo.items()])
        if solve_once:
            pth = os.path.join(profile_fldr, base + f'_once_id={i}_run_time.prof')
            mb.solve_once(method='auto', run_time_profile_file=pth)

        # don't test combinations that would just rerun using the whole set of unadded cuts
        # make sure these still print out correctly
        for min_search_proportion in min_search_proportions:
            if cut_size >= min_search_proportion * cuts_possible[mb.n]:
                continue
            profile_pth = os.path.join(
                profile_fldr, base + f'_msp={min_search_proportion}_id={i}_run_time.prof')
            mb.solve_iteratively(warm_start=True, method='auto',
                                 min_search_proportion=min_search_proportion,
                                 run_time_profile_file=profile_pth)

        for threshold_proportion in threshold_proportions:
            if cut_size >= (1 - threshold_proportion) * cuts_possible[mb.n]:
                continue
            profile_pth = os.path.join(
                profile_fldr, base + f'_tp={threshold_proportion}_id={i}_run_time.prof')
            mb.solve_iteratively(warm_start=True, method='auto',
                                 threshold_proportion=threshold_proportion,
                                 run_time_profile_file=profile_pth)

        solution_schema.csv.write_directory(mb.data, os.path.join(fldr, str(i)),
                                            allow_overwrite=True)


if __name__ == '__main__':
    kwargs = {
        'ns': [int(n) for n in sys.argv[2:]],
        'ps': [.5, .8],
        'qs': [.1, .2],
        'numbers_of_cuts': [10, 100],
        'min_search_proportions': [1],
        # 'threshold_proportions': [.5, .9],
        'repeats': 5,
        'fldr': sys.argv[1],
        'solve_once': True
    }
    run_experiments(**kwargs)
