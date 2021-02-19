"""
Call with python runner.py <fldr> <n_1> ... <n_m>
"""

from itertools import product
import os
import shutil
import sys

from min_bisection import MinBisect, solution_schema, create_constraint_indices


def run_experiments(ns, ps, qs, cut_proportions=None, numbers_of_cuts=None,
                    solve_once=True, min_search_proportions=None,
                    threshold_proportions=None, remove_constraint_list=None,
                    zero_slack_likelihoods=None, repeats=1, fldr='run_results'):
    """Solve all at once and solve iteratively the min bisection problem
    <repeats> times for each combo of n, p, q, and cut_proportion/number_of_cuts.

    :param ns: list of problem sizes to test
    :param ps: list of intra-cluster edge likelihoods to try
    :param qs: list of inter-cluster edge likelihoods to try
    :param cut_proportions: list of cut proportions to try for each n, p, q combo
    :param numbers_of_cuts: list of fixed number of cuts to try for each n, p, q combo
    :param solve_once: True includes single solves in batch runs, False does not.
    :param min_search_proportions: list of smallest proportion of available cuts
    to search for deepest cuts to add to the model when solving iteratively
    :param threshold_proportions: list of proportions of infeasible cuts one cut
    must be more violated than to be added to the model when solving iteratively
    :param repeats: how many times to rerun each n, p, q, cut_proportion/fixed cuts combo
    :param fldr: where in the working directory to save the results of all
    experiments to csv's
    :return:
    """
    assert numbers_of_cuts or cut_proportions, 'at least one'
    numbers_of_cuts = numbers_of_cuts or []
    cut_proportions = cut_proportions or []
    min_search_proportions = min_search_proportions or [1]
    threshold_proportions = threshold_proportions or []
    remove_constraint_list = remove_constraint_list or [False]
    zero_slack_likelihoods = zero_slack_likelihoods or [0]
    assert False not in remove_constraint_list or 0 in zero_slack_likelihoods

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
        base = '_'.join([f'{"nc" if k=="number_of_cuts" else k}={v}' for k, v in
                         combo.items()])

        if solve_once:
            pth = os.path.join(profile_fldr, base + f'_once_id={i}_run_time.prof')
            mb.solve_once(method='auto', run_time_profile_file=pth)

        # continues avoid combinations that reduce to a previously run set
        for remove_constraints in remove_constraint_list:
            for zero_slack_likelihood in zero_slack_likelihoods:
                if not remove_constraints and zero_slack_likelihood:
                    continue

                for min_search_proportion in min_search_proportions:
                    if cut_size >= min_search_proportion * cuts_possible[mb.n]:
                        continue
                    ext = f'msp={min_search_proportion}_rc={remove_constraints}_'\
                          f'zsl={zero_slack_likelihood}_id={i}_run_time.prof'
                    profile_pth = os.path.join(profile_fldr, base + ext)
                    mb.solve_iteratively(warm_start=True, method='auto',
                                         min_search_proportion=min_search_proportion,
                                         remove_constraints=remove_constraints,
                                         zero_slack_likelihood=zero_slack_likelihood,
                                         run_time_profile_file=profile_pth)

                for threshold_proportion in threshold_proportions:
                    if cut_size >= (1 - threshold_proportion) * cuts_possible[mb.n]:
                        continue
                    ext = f'tp={threshold_proportion}_rc={remove_constraints}_'\
                          f'zsl={zero_slack_likelihood}_id={i}_run_time.prof'
                    profile_pth = os.path.join(profile_fldr, base + ext)
                    mb.solve_iteratively(warm_start=True, method='auto',
                                         threshold_proportion=threshold_proportion,
                                         remove_constraints=remove_constraints,
                                         zero_slack_likelihood=zero_slack_likelihood,
                                         run_time_profile_file=profile_pth)

        solution_schema.csv.write_directory(mb.data, os.path.join(fldr, str(i)),
                                            allow_overwrite=True)


if __name__ == '__main__':
    kwargs = {
        'ns': [int(n) for n in sys.argv[2:]],
        'ps': [.5],
        'qs': [.2],
        'numbers_of_cuts': [1000],
        'solve_once': True,
        'min_search_proportions': [1],
        'remove_constraint_list': [False, True],
        'zero_slack_likelihoods': [0, .1, .3],
        'repeats': 50,
        'fldr': sys.argv[1],
    }
    run_experiments(**kwargs)
