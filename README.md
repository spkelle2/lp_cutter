# lp_cutter
This library will eventually serve a method for solving large linear programs
by finding the optimal solution for a small subset of the constraints and then
resolving with incrementally more until all constraints are accounted for.
Currently, it serves as a sandbox for working out the implementation.

To work within the sandbox, clone the git repo and create the conda environment
in `environment.yml`. If you could use some reminders on how to create and start
conda environments, [this](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
is a good resource. If that looks like gibberish to you, contact Sean directly,
and he'll help you get setup. Once you have the conda environment activated
(or you IDE pointed at it), the following file will be of interest to you.

### min_bisection.py
This file implements the work above. Run it with:
```bash
python min_bisection.py <n> <p> <q> <cut_proportion>
```
See the doc strings in `min_bisection.py` for more information on those parameters.

This will run our model against a randomly generated graph in the code. For large
enough graphs, it will add constraints in chunks to avoid solving the whole model
in one go.
