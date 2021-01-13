# lp_cutter
This repo holds an object with methods for solving large instances of the Minimum
Bisection LP relaxation by finding the optimal solution for a small subset of the
constraints and then re-solving with incrementally more until all constraints are
satisfied.

To run the code, clone the git repo and create the conda environment
in `environment.yml`. If you could use some reminders on how to create and start
conda environments, [this](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
is a good resource. If that looks like gibberish to you, contact Sean directly,
and he'll help you get setup. Once you have the conda environment activated
(or you IDE pointed at it), the following files will be of interest to you.

If you would like to contribute, please make a branch off of `dev` and be sure
to augment the unit tests to account for your changes and that they all pass
before opening a pull request.

### min_bisection.py
This module implements the above mentioned work. It has the following two public
instance methods:
* MinBisect.solve_iteratively()
* MinBisect.solve_once()
 
The first method does as the opening sentence describes, re-solving the LP with
the most violated subset of the constraints each iteration until all constraints
are satisfied. The second method throws all constraints into the first solve of
the model, returning the same solution. For more details on each, feel free to
check out the doc strings. Other than constructing the object, there are no
other methods a non-contributing user should need to interface with in this module.

### runner.py
Solves once and iteratively the Min Bisection Problem for each combination of 
the provided values of parameters. This is used for kicking off batch solves
and collecting their corresponding data. 

### graphs.ipynb
A jupyter notebook where run statistics are compiled and fed into graphs for
comparison

### job.pbs
A template file that can be altered to kick off batch jobs (i.e. `runner.py`)
on the coral servers at Lehigh.