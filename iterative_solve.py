import gurobipy as gu
from ticdat import standard_main

from solve import input_schema, solution_schema


def solve(dat):
    assert input_schema.good_tic_dat_object(dat)
    assert not input_schema.find_foreign_key_failures(dat)
    assert not input_schema.find_data_type_failures(dat)
    assert not input_schema.find_data_row_failures(dat)

    mdl = gu.Model("iteratively_solved_diet")

    # Create decision variables for how much of each macro to consume
    nutrition = {c: mdl.addVar(lb=n["Min Nutrition"], ub=n["Max Nutrition"], name=c)
                 for c, n in dat.categories.items()}

    # Create decision variables for the foods to buy
    buy = {f: mdl.addVar(name=f) for f in dat.foods}

    # Minimize the total cost
    mdl.setObjective(gu.quicksum(buy[f] * c["Cost"] for f, c in dat.foods.items()),
                     sense=gu.GRB.MINIMIZE)

    # Nutrition constraints
    for c in dat.categories:
        mdl.addConstr(gu.quicksum(dat.nutrition_quantities[f, c]["Quantity"] * buy[f]
                                  for f in dat.foods) == nutrition[c], name=c)

        mdl.optimize()

        if mdl.status == gu.GRB.OPTIMAL:
            sln = solution_schema.TicDat()
            for f, x in buy.items():
                if x.x > 0:
                    sln.buy_food[f] = x.x
            for k, x in nutrition.items():
                sln.consume_nutrition[k] = x.x
            sln.parameters['Total Cost'] = sum(dat.foods[f]["Cost"] * r["Quantity"]
                                               for f, r in sln.buy_food.items())
            print(sln)
        else:
            print('something went wrong here.')

    return sln


if __name__ == "__main__":
    standard_main(input_schema, solution_schema, solve)
