from functools import partial, update_wrapper


def register_tasks_and_solvers(task, task_configs, solver_configs):
    """
    Eventually, this will register with the testing system
    that `task : Task` can be run with the given configurations
    on the given solvers.
    (Then the system, when asked to run each solver on each task,
    will include this task with these configurations and these solvers.)

    Args:
    - task : Task
    - task_configs : List[Dict[str, Any]].
        Each dict should have fields:
        - "name" : str (name for this config)
        - "task_args" : dict (named args to pass to the task's constructor)
    - solver_configs : List[Dict[str, Any]]
        Each dict should have fields:
        - "name" : str (name for this config)
        - "solver" : function (solver for the task)
            signature: task_input, *args -> task_solution
        - "solver_args" : dict of kwargs to pass to the solver
            after the task_input
    """
    # TODO:Karen - will this always work,
    # or do we need something fancier?

    tasks = [
        task(**config['task_args'])
        for config in task_configs
    ]
    solvers = [
        update_wrapper(partial(solver_config['solver'],**solver_config['solver_args']), solver_config['solver'])
        for solver_config in solver_configs  # functools hack to preserve solver function name for JSON export
    ]
    return (tasks, solvers)
