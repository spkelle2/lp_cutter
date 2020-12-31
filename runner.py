"""
Call with python runner.py <fldr> <n_1> ... <n_m>
"""

from itertools import product
import sys

from min_bisection import MinBisect, solution_schema


# since they're random, we could maybe have once over solves go on their own
def run_experiments(ns, ps, qs, cut_proportions=None, numbers_of_cuts=None,
                    repeats=1, warm_starts=None, methods=None, fldr='run_results'):
    """Solve all at once and solve iteratively the min bisection problem
    <repeats> times for each combo of n, p, q, and cut_proportion/number_of_cuts.

    :param ns: list of problem sizes to test
    :param ps: list of intra-cluster edge likelihoods to try
    :param qs: list of inter-cluster edge likelihoods to try
    :param cut_proportions: list of cut proportions to try for each n, p, q combo
    :param numbers_of_cuts: list of fixed number of cuts to try for each n, p, q combo
    :param repeats: how many times to rerun each n, p, q, cut_proportion/fixed cuts combo
    :param warm_starts: list of booleans representing whether or not to warm start
    iterative solves
    :param methods: list of strings of solve methods to try
    :param fldr: where in the working directory to save the results of all
    experiments to csv's
    :return:
    """
    numbers_of_cuts = [] if not numbers_of_cuts else numbers_of_cuts
    cut_proportions = [] if not cut_proportions else cut_proportions
    warm_starts = [True] if not warm_starts else warm_starts
    methods = ['dual'] if not methods else methods

    output = solution_schema.TicDat()
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
        print(f'running test {i} of {len(combinations)}')
        mb = MinBisect(**combo, solve_id=i, tolerance=0)
        for method in methods:
            mb.solve_once(method)
            for warm_start in warm_starts:
                mb.solve_iteratively(warm_start, method)
            for t in solution_schema.all_tables:
                for pk, f in getattr(mb.data, t).items():
                    getattr(output, t)[pk] = f
            solution_schema.csv.write_directory(output, fldr, allow_overwrite=True)


if __name__ == '__main__':
    kwargs = {
        'ns': [int(n) for n in sys.argv[2:]],
        'ps': [.5, .8],
        'qs': [.1, .2],
        'numbers_of_cuts': [10, 30, 100, 300, 1000],
        'repeats': 3,
        'warm_starts': [True, False],
        'methods': ['dual', 'auto'],
        'fldr': sys.argv[1]
    }
    run_experiments(**kwargs)

