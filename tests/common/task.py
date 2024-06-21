class Task:
    """
    Lightweight base class for defining tasks to be solved by parts of the b3d codebase.
    """

    ### CORE TASK INTERFACE ###
    ### Task developers should define the following. ###

    def get_task_specification() -> "Tuple[Any, Any]":
        """
        Outputs `task_spec`, the input given to a task solver.
        """
        raise NotImplementedError()
    
    def score(self, solution) -> "Any":
        """
        Scores an attempted solution ot the task, returning a collection of metrics.

        Args:
            - `solution: Any`.  The output of the solver, which is to be scored.

        Returns:
            - `metrics: Any`.  A python dictionary (which can be JSONified) containing
                a collection of metrics measuring how well the given `solution` solves the task.
        """
        raise NotImplementedError()
    
    def assert_passing(self, scores, *args, **kwargs) -> None:
        """
        Takes the output of `score` and makes assertions about the scores,
        asserting that the scores represent a solution
        to the task which is acceptable.

        Args:
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
    
    ### Optional Viz Methods ###
    def visualize_task(self):
        """
        Visualize the task (but not the solution).
        This may log data to rerun, produce pyplots, etc.
        """
        raise NotImplementedError()
    
    def visualize_solution(self, solution, metrics):
        """
        Visualize a solution to the task.
        This may log data to rerun, produce pyplots, etc.
        """
        raise NotImplementedError()

    ### Methods automatically implemented from the above. ###

    def run_and_score(self, solver, viz=False) -> dict:
        """
        Solve the task with `solver` and score the result.

        Args:
            solver: Solver object with a `solve` method.

        Returns:
            metrics: A dictionary containing the results of the test.

        This is implemented automatically from `self.get_test_pair` and `self.score`.
        """
        task_spec = self.get_task_specification()
        if viz:
            self.visualize_task()
        task_output = solver.solve(task_spec)
        metrics = self.score(task_output)
        if viz:
            self.visualize_solution(task_output, metrics)
        return metrics

    def run_tests(self, solver, *args, viz=False, **kwargs) -> None:
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
        metrics = self.run_and_score(solver, viz=viz)
        self.assert_passing(metrics, *args, **kwargs)
