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
(or you IDE pointed at it), the following two files will be of interest to you.

### solve.py
This file is the diet problem implemented with ticdat's ORM and solved with gurobi
([what is ticdat](https://ticdat.github.io/ticdat/) and
[oh, hey, tell me more](https://github.com/ticdat/ticdat/wiki)). Nothing fancy
happening here. Feel free to run it with the following command:

```bash
python solve.py -i diet_sample_data -o normal_out.xlsx
```

This will run the diet problem against the csv's in `diet_sample_data` and
save them to the specified excel spreadsheet. Until we have some unit tests,
this will serve to check to make sure our output to the next file is correct.

### iterative_solve.py
This file is again the diet problem implemented in ticdat and solved with gurobi.
This time, however, the solve is done iteratively by feeding a larger and larger
nested subset of the constraints to gurobi. Feel free to run it with the
following command:

```bash
python iterative_solve.py -i diet_sample_data -o iterative_out.xlsx
```

Like the above, this will run the given problem against the `diet_sample_data`
csv's and save the results of the final model run to the given excel file.