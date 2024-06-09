# task_tests

This directory contains tests of the ChiSight system at solving tasks (such as perception tasks).  The code in this directory fulfills a role somewhere between the role of "integration testing" of an end-to-end engineering system, and the role of conducting an evaluation of a machine learning system.

(See `tests/unit_tests` for tests of smaller components of the system.)

## Running the tests

Option 1: Run the test-running script.  Run `./runtests.sh`
from within the `task_tests` directory.

Option 2: Run the tests as python scripts in interactive mode. Run
```bash
python -i -m tests.test_X
```
from the `task_tests` directory, where `X` is the name of the suffix of the
test file you want to run.
(This will _not_ run the tests in the file, but will instead drop you into an
interactive python session where you can run the tests manually.)

Option 3: Run the test-running script for a single test. Run
```bash
python -m unittest tests.text_X -v
```
from the `task_tests` directory, where `X` is the name of the suffix of the
test file you want to run.
(This will run the tests in the file, and will print the results to the console, similarly
to the result from `./runtests.sh`.)


## Directory structrure
- `tasks`: Declarations of `Task` objects, which define tasks that the system should be able to solve.
- `solvers`: Implementations of "solvers" for the `Task`s.
- `tests`: Files which pair together tasks with the solvers capable of solving them.
- `common`: This primarily contains the implementation of the testing and metrics system.
    Among other things it includes a definition of the `Task` class, in `task.py`.

## Tasks
See [common/task.py](common/task.py) for the definition of the `Task` class.
The docstrings explain how to define a new task, by defining a new subclass of `Task`.
This should be done in a python file in the [tasks](tasks) directory.

A Task developer must implement
1. `Task.__init__(**config)`, which instantiates a particular instance of the task
    from the provided config.  The config may specify, for instance, a video file
    to process.  (The instantiated task is then to process that particular video.)
2. `Task.get_test_pair()`, which returns a pair `(task_input, baseline)`.
    `task_input` is the input to a task solver. `baseline` is additional information
    which might be used to score the solution produced by a solver.
3. `Task.score(task_input, baseline, solution)` outputs a JSON-able python data
    structure containing a collection of metrics, scoring how good a solution to the task
    the provided `solution` is.
4. `Task.assert_passing(scores, *args, **kwargs)` returns a boolean, indicating whether
    the scores are good enough for the solution that resulted in them to be considered
    a viable solution to the task.

## Solvers

Once a task is defined, a solver is simply a python function with signature `task_input -> solution`,
where `task_input` is of the type produced by `Task.get_test_pair`, and solution is of the type
expected by `Task.score`.

Solvers will often be implemented by wrapping code from the main DCOLMAP and HGPS codebases
as to satisfy the input/output interface in a way expected by the `Task`.

Solvers will go in the [solvers](solvers) subdirectory.

We also included a directory [tasks/utils](tasks/utils), which we envision may be useful for including code useful for solving different tasks, but which is not ready to go in the main DCOLMAP or HGPS codebases.

We currently envision that solvers will be organized in subdirectories of [solvers](solvers), with one subdirectory per task, and each directory will contain solvers for that task.
Currently the system does not rely on this structure, however.

## Tests

Once tasks are developed, and solvers are implemented, the [tests](tests) subdirectory is used to
1. Declare to the metrics & testing system, for each task:
    1. A list of configurations for that task.  (That is, a list of arguments to the Task constructor.  These specify, for instance, all of the data files the Task can be constructed from.  This may be constructed programmatically.)
    2. A list of solvers for that task.  (This can be constructed by providing a small list of solver functions which themselves require a configuration to specify hyperparameters, and a larger list of hyperparameters for each solver.)
2. Write tests which will be triggered when the continuous integration system runs, to detect regressions in the performance of the solvers.

Step (1) will result in the metrics and testing system (implemented in [common](common)) having a list of tasks, and for each task, a list of configurations and a list of solvers.  We will ultimately develop scripts which can run each solver on each task configuration, score the results for each, and export all the results to JSON files and plots.  We will also develop other export modes, such as exporting a grid of which solvers "passed" each task, according to `Task.assert_passing`.  We also intend to eventually allow Tasks to declare visualizations which they can produce given any solution, and have a script to automatically export all visualizations for all solutions to all tasks.

Step (2) is used to raise errors and warnings when the behavior of solvers on tasks falls outside the expected range.  Note that this is may be different from the pass/fail behavior designated by `Task.assert_passing`, which is supposed to indicate if a solution is a viable solution to a task.  We currently anticipate that we will maintain solvers which are baselines to the tasks, and produce poor solutions.  We will want to see that these baselines are poor solutions when we export metrics or a "pass/fail" grid, but we will not want the continuous integration system to raise errors and warnings due to their poor behavior.

We envision that often, Step (2) will be implemented by calling `Task.passes_tests(solver, *args, **kwargs)`, using the `args` and `kwargs` to pass in non-default thresholds to use to test the scores produced by `Task.score` on the solutions generated by the `solver`.