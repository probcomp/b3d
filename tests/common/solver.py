class Solver:
    """
    Lightweight base class for defining solvers for `Task`s (see `tests/common/task.py`).

    One design pattern I have found useful is to have the `Solver.solve` method store
    state on the `Solver` (self) object that is useful for visualizing the internal state
    of the solver (e.g. genjax traces that were produced during inference).
    The `visualize_solver_state` method can be used to visualize this state (e.g. by logging
    it to rerun).
    """

    def solve(self, task_specification):
        """
        Accepts as input the `task_specification` returned
        by `Task.get_task_specification` and returns a solution to the task
        accepted by `Task.score`.
        """
        raise NotImplementedError()

    def visualize_solver_state(self):
        """
        Visualize any information recorded by the solver during the last call to `solve`.
        This may log data to rerun, produce pyplots, etc.
        """
        pass

    @property
    def name(self) -> str:
        """
        Returns the name of the solver.
        """
        return self.__class__.__name__
