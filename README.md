# lp_cutter
This repo holds an object with methods for solving large instances of the Minimum
Bisection LP relaxation by finding the optimal solution for a small subset of the
constraints and then resolving with incrementally more until all constraints are
satisfied.

To run the code, clone the git repo and create the conda environment
in `environment.yml`. If you could use some reminders on how to create and start
conda environments, [this](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
is a good resource. If that looks like gibberish to you, contact Sean directly,
and he'll help you get setup. Once you have the conda environment activated
(or you IDE pointed at it), the following file will be of interest to you.

If you would like to contribute, please make a branch off of `dev` and be sure
to augment the unit tests to account for your changes and that they all pass
before opening a pull request.

### min_bisection.py
This file implements the work above. Run it with:
```bash
python min_bisection.py <n> <p> <q> <cut_proportion>
```
See the doc strings in `min_bisection.py` for more information on those parameters.

