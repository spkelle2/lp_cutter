from ticdat import TicDatFactory
from ticdat.utils import standard_main

input_schema = TicDatFactory(
    parameters=[['Name'], ['Value']],
    a='*'
)
input_schema.add_parameter("Cut Proportion", default_value=.1, min=0, max=1)

solution_schema = TicDatFactory(
    summary=[['Name'], ['Value']],
    x=[['Node'],['Cluster']]
)


def solve(dat):
    print()


if __name__ == '__main__':
    standard_main(input_schema, solution_schema, solve)