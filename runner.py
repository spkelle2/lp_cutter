"""
Call with python runner.py <fldr> <n_1> ... <n_m>
"""

from itertools import product
import sys

from min_bisection import MinBisect, solution_schema, create_constraint_indices


# since they're random, we could maybe have once over solves go on their own
def run_experiments(ns, ps, qs, cut_proportions=None, numbers_of_cuts=None,
                    min_orders=None, repeats=1, fldr='run_results'):
    """Solve all at once and solve iteratively the min bisection problem
    <repeats> times for each combo of n, p, q, and cut_proportion/number_of_cuts.

    :param ns: list of problem sizes to test
    :param ps: list of intra-cluster edge likelihoods to try
    :param qs: list of inter-cluster edge likelihoods to try
    :param cut_proportions: list of cut proportions to try for each n, p, q combo
    :param numbers_of_cuts: list of fixed number of cuts to try for each n, p, q combo
    :param min_orders: list of minimum order of magnitude larger search set must
    be than desired selected cut set
    :param repeats: how many times to rerun each n, p, q, cut_proportion/fixed cuts combo
    :param fldr: where in the working directory to save the results of all
    experiments to csv's
    :return:
    """
    numbers_of_cuts = [] if not numbers_of_cuts else numbers_of_cuts
    cut_proportions = [] if not cut_proportions else cut_proportions
    min_orders = [0] if not min_orders else min_orders

    cuts_possible = {n: len(create_constraint_indices(range(n))) for n in ns}

    output = solution_schema.TicDat()

    # don't test combinations with min order that would just rerun using the
    # whole set of unadded cuts
    combinations = [
        {'n': n, 'p': p, 'q': q, 'cut_proportion': cut_proportion,
         'min_order': min_order} for (n, p, q, cut_proportion, min_order) in
        product(ns, ps, qs, cut_proportions, min_orders) if min_order == 0 or
        10**min_order*cut_proportion*cuts_possible[n] < cuts_possible[n]
    ] + [
        {'n': n, 'p': p, 'q': q, 'number_of_cuts': number_of_cuts,
         'min_order': min_order} for (n, p, q, number_of_cuts, min_order) in
        product(ns, ps, qs, numbers_of_cuts, min_orders) if min_order == 0 or
        10**min_order*number_of_cuts < cuts_possible[n]
    ]
    combinations *= repeats
    combinations.sort(key=lambda x: x['n'])

    for i, combo in enumerate(combinations):
        print(f'running test {i+1} of {len(combinations)}')
        mb = MinBisect(**combo, solve_id=i, tolerance=0)
        mb.solve_once(method='auto')
        mb.solve_iteratively(warm_start=True, method='dual')
        for t in solution_schema.all_tables:
            for pk, f in getattr(mb.data, t).items():
                getattr(output, t)[pk] = f
        solution_schema.csv.write_directory(output, fldr, allow_overwrite=True)


if __name__ == '__main__':
    kwargs = {
        'ns': [int(n) for n in sys.argv[2:]],
        'ps': [.5, .8],
        'qs': [.1, .2],
        'numbers_of_cuts': [10, 30, 100, 300, 1000, 3000, 10000],
        'min_orders': [0, 1, 2, 3],
        'repeats': 5,
        'fldr': sys.argv[1]
    }
    run_experiments(**kwargs)

