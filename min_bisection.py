import gurobipy as gu

from ticdat import TicDatFactory
from ticdat.utils import standard_main

input_schema = TicDatFactory(
    parameters=[['Name'], ['Value']],
    a=[['Vertex 1', 'Vertex 2'], ['Exists']]
)

input_schema.set_data_type('a', 'Vertex 1', must_be_int=True)
input_schema.set_data_type('a', 'Vertex 2', must_be_int=True)
input_schema.set_data_type('a', 'Exists', inclusive_max=True, max=1,
                           must_be_int=True)

# dont repeat edges and no edge starts and ends at same vertex
input_schema.add_data_row_predicate(
    "a", predicate_name="Allowable Edge Check",
    predicate=lambda row: row["Vertex 1"] < row["Vertex 2"])

input_schema.add_parameter("Cut Proportion", default_value=.1, inclusive_max=True,
                           min=0, max=1)

solution_schema = TicDatFactory(
    summary=[['Name'], ['Value']],
    x=[['Node'], ['Cluster']]
)


def data_integrity_checks(dat):
    """Check that the data for our min bisection problem is "good"

    :param dat: the ticdat to check
    :return:
    """
    assert input_schema.good_tic_dat_object(dat)
    assert not input_schema.find_data_type_failures(dat)
    assert not input_schema.find_data_row_failures(dat)

    indices = range(max(v2 for (v1, v2) in dat.a) + 1)
    missing_pairs = [(i, j) for i in indices[:-1] for j in indices[i+1:] if
                     dat.a.get((i, j)) is None]
    assert not missing_pairs, f'Provide values for these vertex pairs: {missing_pairs}'


def solve(dat):
    """A model for solving min bisection set up such that it is run repeatedly,
    adding the most violated constraints each iteration until all constraints
    are satisfied.

    :param dat: a ticdat representing our input data for the min bisection model
    :return:
    """
    data_integrity_checks(dat)

    mdl = gu.Model("min bisection")

    x = {(v1, v2): mdl.addVar(vtype='B', name=(v1, v2)) for (v1, v2) in dat.a}

    mdl.setObjective(gu.quicksum(f['Exists'] * x[v1, v2] for (v1, v2), f in dat.a),
                     sense=gu.GRB.MINIMIZE)


if __name__ == '__main__':
    standard_main(input_schema, solution_schema, solve)