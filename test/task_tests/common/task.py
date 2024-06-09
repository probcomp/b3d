from .common import TestCase

class Task:
    """
    This is the base class for tasks that the ChiSight perception systen is expected
    to solve.

    To define a task, one subclasses `Task`, and implements:
    - `get_test_pair()` which returns a pair `(task_input, baseline)`.
        - `task_input` will be given to a task solver to produce a solution.
        - `baseline` will be used to score the solution.
    - `score(task_input, baseline, solution)` which returns a collection of metrics measuring
        how well the given `solution` solves the task.
    - [Optional] `assert_testing(tester: TestCase, scores, *args, **kwargs)` which takes the data structure of metrics
        output by `score`, and makes assertions testing that the metrics reflect
        an acceptable solution to the task.  (`tester: TestCase` is a `unittest.TestCase` object.
        Its methods should be used to make assertions about the scores.)

    The constructor for the Task may be used to specify the task configuration
    (including task hyperparameters, data files the task should be run on, etc.).

    Given a `solver`, a function from task_input -> solution, one may run the following
    methods.
    - `task.run_and_score(solver)` to solve the task and score the solution.
    - `task.run_tests(tester: TestCase, solver)` to test whether the given `solver` produces an acceptable
        solution to the task.
    """

    ### Task developers should define the following. ###

    def get_test_pair() -> "Tuple[Any, Any]":
        """
        Outputs a pair (task_input, baseline).

        Return values:
            - `task_input: Any`.  The input given to task solvers, to produce a solution.
            - `baseline: Any`.  A python value containing any other information which is needed to score the solutions.
                (For instance, the `baseline` may be a "ground truth" solution, so the solver's solution
                can be scored in terms of its error from ground truth.)
        """
        raise NotImplementedError()
    
    def score(self, task_input, baseline, solution) -> "Any":
        """
        Scores an attempted solution ot the task, returning a suite of metrics.

        Args:
            - `task_input: Any`.  `task_input` from `get_test_pair()`
            - `baseline: Any`.  `baseline` from `get_test_pair()`
            - `solution: Any`.  The output of the solver, which is to be scored.

        Returns:
            - `metrics: Any`.  A python dictionary (which can be JSONified) containing
                a collection of metrics measuring how well the given `solution` solves the task.
        """
        raise NotImplementedError()
    
    def assert_testing(self, tester: TestCase, scores, *args, **kwargs) -> None:
        """
        Takes the output of `score` and makes assertions about the scores,
        asserting that the scores represent a solution
        to the task which is acceptable.
        
        Assertions should be raised by calling assertion methods on the provided
        `tester: TestCase` object (e.g. `tester.assertEqual(a, b)`).
        (`TestCase` inherits from `unittest.TestCase`, so it has all the same assertion methods
        as are available in the unittest library.
        Useful resource for available assertions:
        https://kapeli.com/cheat_sheets/Python_unittest_Assertions.docset/Contents/Resources/Documents/index
        )

        Args:
            - `tester: TestCase`.  A `unittest.TestCase` object, which can be used to
                make assertions about the scores (by calling things like `tester.assertEqual(a, b)`.)
            - `scores: Any`.  The output of `Task.score`.
            - `*args, **kwargs`.  Optional arguments which can be used to change the testing behavior.

        This function, if implemented, should run when `task.assert_testing(tester, scores)` is called.
        It may also accept optional *args and **kwargs to change the testing behavior.
        (This can be used when declaring the continuous integration mode for tests.
        There, one may want to run tests with different tolerances,
        so warnings are not thrown by continuous integration on solvers known to fail the task
        according to the default tolerances).
        """
        raise NotImplementedError()

    def export_visuals(self, task_input, baseline, solution):
        """
        Currently not supported.  Eventually, this function will be added, enabling
        Task developers to develop a suite of visualizations which can automatically
        be produced from any solution to the task, and saved to disk as mp4 files, gifs, etc.

        (Task developers can still implement this if they'd like, and call it manually
        from their test scripts.  But it won't be called automatically by the testing system.)
        """
        pass

    def setup_viser(self, server, task_input, baseline, solution):
        """
        Currently not supported.  Eventally, this function will be added,
        enabling Task developers to set up a default suite of visalizations
        which can be loaded into Viser, for any solution to this task.

        (Task developers can still implement this if they would like, and call it manually
        from their test scripts.  But it won't be called automatically by the testing system.)
        """
        pass
    

    ### Methods automatically implemented from the above. ###

    def run_and_score(self, solver) -> dict:
        """
        Solve the task with `solver` and score the result.

        Args:
            solver: Function with signature `task_input -> solution`

        Returns:
            metrics: A dictionary containing the results of the test.

        This is implemented automatically from `self.get_test_pair` and `self.score`.
        """
        task_input, baseline = self.get_test_pair()
        task_output = solver(task_input)
        return self.score(task_input, baseline, task_output)

    def run_tests(self, tester: TestCase, solver, *args, **kwargs) -> None:
        """
        Solve the task with `solver` and test that the resulting metrics measuring solution quality
        indicate that the solution is acceptable.
        (If it is not acceptable, assertions will be raised from the `tester: TestCase` object).

        Args:
            solver: Function with signature `task_input -> solution`.

        This is implemented automatically from `self.get_test_pair`, `self.assert_passing` , and `self.score`.
        *args and **kwargs are passed to `self.assert_passing` (e.g. to configure tolerances for the tests applied
        to the metrics).
        """
        # Score and assess whether passing.
        metrics = self.run_and_score(solver)
        
        tester.prepare_json_export(solver, metrics, *args, **kwargs)        
        self.assert_passing(tester, metrics, *args, **kwargs)
