from itertools import product

from min_bisection import MinBisect, solution_schema


# since they're random, we could maybe have once over solves go on their own
def run_experiments(ns, ps, qs, cut_proportions=None, numbers_of_cuts=None, repeats=1,
                    fldr='run_results'):
    """Solve all at once and solve iteratively the min bisection problem
    <repeats> times for each combo of n, p, q, and cut_proportion/number_of_cuts.

    :param ns: list of problem sizes to test
    :param ps: list of intra-cluster edge likelihoods to try
    :param qs: list of inter-cluster edge likelihoods to try
    :param cut_proportions: list of cut proportions to try for each n, p, q combo
    :param numbers_of_cuts: list of fixed number of cuts to try for each n, p, q combo
    :param repeats: how many times to rerun each n, p, q, cut_proportion/fixed cuts combo
    :param fldr: where in the working directory to save the results of all
    experiments to csv's
    :return:
    """
    numbers_of_cuts = [] if not numbers_of_cuts else numbers_of_cuts
    cut_proportions = [] if not cut_proportions else cut_proportions

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
        mb = MinBisect(**combo, solve_id=i)
        mb.solve_once()
        mb.solve_iteratively()
        for t in solution_schema.all_tables:
            for pk, f in getattr(mb.data, t).items():
                getattr(output, t)[pk] = f
        solution_schema.csv.write_directory(output, fldr, allow_overwrite=True)


if __name__ == '__main__':
    kwargs = {
        'ns': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'ps': [.5],
        'qs': [.1, .2],
        'cut_proportions': [.1],
        'repeats': 3
    }
    run_experiments(**kwargs)

