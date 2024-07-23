from functools import wraps
from warnings import warn


def depreciated(msg):
    """
    Decorator to mark functions as depreciated.

    Example Usage:
    ```
    @depreciated("Use `something_new` instead.")
    def something_old(x):
        return x
    ```

    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            warn(
                f"Call to DEPRECIATED func `{f.__name__}`...{msg}",
                DeprecationWarning,
                stacklevel=2,
            )
            return f(*args, **kwargs)

        return wrapper

    return decorator
