"""
Call with python runner.py <fldr> <n_1> ... <n_m>
"""

from itertools import product
import sys

from min_bisection import MinBisect, solution_schema, create_constraint_indices


# since they're random, we could maybe have once over solves go on their own
def run_experiments(ns, ps, qs, cut_proportions=None, numbers_of_cuts=None,
                    min_search_proportions=None, threshold_proportions=None,
                    repeats=1, fldr='run_results'):
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
    :return:
    """
    numbers_of_cuts = [] if not numbers_of_cuts else numbers_of_cuts
    cut_proportions = [] if not cut_proportions else cut_proportions
    min_search_proportions = [1] if not min_search_proportions else min_search_proportions
    threshold_proportions = [] if not threshold_proportions else threshold_proportions

    cuts_possible = {n: len(create_constraint_indices(range(n))) for n in ns}

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
        print(f'running test {i+1} of {len(combinations)}')
        mb = MinBisect(**combo, solve_id=i)
        cut_size = mb.cut_value if mb.cut_type == 'fixed' else \
            int(mb.cut_value * cuts_possible[mb.n])
        mb.solve_once(method='auto')

        # don't test combinations that would just rerun using the whole set of unadded cuts
        for min_search_proportion in min_search_proportions:
            if cut_size >= min_search_proportion * cuts_possible[mb.n]:
                continue
            name = '_'.join([fldr] + [f'{k}={v}' for k, v in combo.items()] +
                            [f'min_search_proportion={min_search_proportion}_solve_id={i}.prof'])
            mb.solve_iteratively(warm_start=True, method='auto',
                                 min_search_proportion=min_search_proportion,
                                 output_file=name)

        for threshold_proportion in threshold_proportions:
            if cut_size >= (1 - threshold_proportion) * cuts_possible[mb.n]:
                continue
            name = '_'.join([fldr] + [f'{k}={v}' for k, v in combo.items()] +
                            [f'threshold_proportion={threshold_proportion}_solve_id={i}.prof'])
            mb.solve_iteratively(warm_start=True, method='auto',
                                 threshold_proportion=threshold_proportion,
                                 output_file=name)

        for t in solution_schema.all_tables:
            for pk, f in getattr(mb.data, t).items():
                getattr(output, t)[pk] = f
        solution_schema.csv.write_directory(output, fldr, allow_overwrite=True)


if __name__ == '__main__':
    kwargs = {
        'ns': [int(n) for n in sys.argv[2:]],
        'ps': [.5, .8],
        'qs': [.1, .2],
        'numbers_of_cuts': [1000, 10000, 100000, 1000000],
        'min_search_proportions': [.01, .1],
        'threshold_proportions': [.5, .9],
        'repeats': 5,
        'fldr': sys.argv[1]
    }
    run_experiments(**kwargs)
