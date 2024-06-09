import collections
import dataclasses
from datetime import datetime
import functools
import inspect
import itertools
import json
import multiprocessing
import numbers
import os
import pathlib
import random
import time
import timeit
import unittest

import jax 
import jax.numpy as jnp 
import numpy as np

from .config import Config

import absl.logging
absl.logging.set_verbosity(absl.logging.WARNING)
from absl.testing import parameterized as atp


current_process = multiprocessing.current_process()
if current_process.name == 'MainProcess':
    print(f'seed = {Config.seed}')
    print(f'device = {jax.default_backend()}')
    print(f'results (JSON) save dir = {Config.save_dir}')

class TestCase(atp.TestCase):
    r'''
    Base class for HGPS-DColmap unit tests.

    Note that this ultimately inherits from `unittest.TestCase`,
    so all the usual `unittest` methods are available.

    One useful resource for the available TestCase assertion methods
    (e.g. `TestCase.assertEqual`, `TestCase.assertAlmostEqual`, etc.) is the following:
    https://kapeli.com/cheat_sheets/Python_unittest_Assertions.docset/Contents/Resources/Documents/index
    '''
    @classmethod
    def setUpClass(self):
        r'''
        Automatically resets random seed to the value provided in config before
        running each test case; 
        Configures directory to save JSON results into and initializes JSON frame
        '''
        
        ## save test run jsons into `task_tests/results/test_current_time_stamp/TestClassName.json`
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir_for_run = Config.save_dir / f"test_{timestamp}"
        os.makedirs(self.save_dir_for_run, exist_ok=True)
        
        ## JSON setup
        self.json_log = {
            'TaskTest': None,
            'results': []
        }
        
    def setUp(self):
        # get name of current Test class 
        self.test_class_name = self.__class__.__qualname__
        
        # log name of json file based on TestClass name
        if self.json_log['TaskTest'] is None:
            self.json_log['TaskTest'] = self.__class__.__qualname__
        self.json_updated = False  # indicates whether current subtest was logged into json.

    
    @classmethod
    def tearDownClass(self):
        """
        At the end of all tests for the TaskTest, write the json log into file.
        """
        self.json_file = self.save_dir_for_run /  f"{self.json_log['TaskTest']}.json"
        with open(self.json_file,'w') as f:
            f.write(
                json.dumps(self.json_log)
            )     
        
    def tearDown(self):
        """
        At end of each subtest, update the json log based on success status,
        if the json log was updated via self.prepare_json_export.
        """
        result = self._outcome.result
        ok = all(test != self for test, text in result.errors + result.failures)
            
        ## Finalize json log based on results.
        if self.json_updated:
            self.json_log['results'][-1]['status'] = "success" if ok else "failure"
            self.json_log['results'][-1]['stderr'] = "" if ok else result.failures[0][1]  
        else:
            self.json_log['results'].append(
                {
                    'name': self._testMethodName,
                    'status': "success" if ok else "failure",
                    'stderr': "" if ok else result.failures[0][1]                          
                }
            )
            absl.logging.warning(f"Full test settings will NOT be logged for {self}...\
                                 check that `TestCase.prepare_json_export` was run as part of your workflow\
                                 to document the solver name and result metrics obtained from the solver.") 

    def __str__(self):
        cls = self.__class__
        return f'{cls.__module__}.{cls.__qualname__}.{self._testMethodName}'

    def assertCorrectness(
        self,
        test_fn,
        ref_fn,
        kwargs,
        extra_test_kwargs=None,
        extra_ref_kwargs=None,
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
        multiple_outputs=False,
    ):
        r'''
        Check that a specified test function matches the reference.

        Relative and absolute tolerance for comparing gradients of inputs w.r.t
        the reference can be customized for each input tensor.

        Args:
            test_fn: test function
            ref_fn: reference function
            kwargs: keyword arguments common for both test and ref function
            extra_test_kwargs: optional extra keyword arguments for test
                function
            extra_ref_kwargs: optional extra keyword arguments for reference
                function
            gradient: optional input gradient w.r.t. the first output of
                functions
            rtol: relative tolerance for output of forward, and default
                relative tolerance for gradients of inputs
            atol: absolute tolerance for output of forward, and default
                absolute tolerance for gradients of inputs
            equal_nan: if ``True``, then two ``NaNs`` will be considered equal
            check_stride: if ``True``, then check strides of all compared
                tensors
            multiple_outputs: if ``True`` then model has multiple outputs which
                will be iterated through to confirm correctness.  Can only
                be used on Forward, will error if inputs require gradient.
        '''
        passed = True
        all_msgs = []
        failed_msgs = []

        if extra_test_kwargs is None:
            extra_test_kwargs = {}

        if extra_ref_kwargs is None:
            extra_ref_kwargs = {}

        fn_kwargs = {
            k: v.tensor if isinstance(v, TestParam) else v
            for k, v in kwargs.items()
        }

        test_out = test_fn(**fn_kwargs, **extra_test_kwargs)
        ref_out = ref_fn(**fn_kwargs, **extra_ref_kwargs)
        if multiple_outputs:
            assert len(test_out) == len(ref_out)
            out_close_all = []
            for to, ro in zip(test_out, ref_out):
                out_close, msg = compare_tensors(
                    to, ro, rtol, atol, equal_nan, 
                )
                out_close_all.append(out_close)
                all_msgs.extend(msg)
            out_close = all(out_close_all)
        else:
            out_close, msg = compare_tensors(
                test_out, ref_out, rtol, atol, equal_nan, 
            )
        if not out_close:
            passed = False
            prefix = (
                '*** OUTPUT DID NOT MATCH THE REFERENCE '
                f'(rtol={rtol}, atol={atol}) ***'
            )
            failed_msgs.append(prefix)
            failed_msgs.extend(msg)
        else:
            prefix = (
                '*** OUTPUT MATCHED THE REFERENCE '
                f'(rtol={rtol}, atol={atol}) ***'
            )
        all_msgs.append(prefix)
        all_msgs.extend(msg)

        all_msgs = '\n'.join([f'\t{msg}' for msg in all_msgs])
        failed_msgs = '\n'.join([f'\t{msg}' for msg in failed_msgs])

        self.assertTrue(passed, f'\n{failed_msgs}')

        if Config.print_matching:
            test = self._subtest if self._subtest is not None else self
            # if not Config.verbose:
            #     print(test)
            print(all_msgs)

    def assertAllClose(
        self,
        input,
        reference,
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
        check_stride=True,
    ):
        raise NotImplementedError("Implement assertAllClose")

    def assertInRange(
        self, input, min:float=-jnp.inf, max:float=jnp.inf
    ):
        assert jnp.all(input >= min) and jnp.all(input <= max), f"{input} Not in range [{min}, {max}]"
        
    ## TODO add more Dcolmap/HGPS project-specific asserts as needed!
    ## e.g. `assertEfficiency(fn_to_benchmark, runtime_threshold)`
    
    def prepare_json_export(self, solver, metrics, *tasktest_args, **tasktest_kwargs):
        """
        For the current subtest, add a new entry in the json log indicating 
        the subtest name, solver name, result metrics (name:value dict), and any arg/kwargs that will be
        passed into the assert-raising metrics function name.
        """
        self.json_log['results'].append({
                        'name': self._testMethodName,
                        'solver': solver.__name__,
                        'metrics': serialize_results_dict(metrics),
                        'args_kwargs': (tasktest_args, tasktest_kwargs),
                    })
        self.json_updated = True

############################
# Export (JSON, TODO image/video?) utils.
############################
def serialize_results_dict(d:dict)->list:
    """
    Flatten dictionary into json-exportable dict format.
    Serializable types in Python are: 
        dictionaries, lists, strings, integers, floats, booleans, None
    
    Args:
    d (dict): {'metric_name': output_object} 
    
    Returns: 
    new_d (dict): {'metric_name': flattened_output_object}
    """
    primitive_types = (numbers.Number, bool, str)
    
    new_d = {}
    for k, v in d.items():
        if isinstance(v, primitive_types) or v is None:
            new_v = v
        elif isinstance(v, dict):
            new_v = serialize_results_dict({v})
        elif isinstance(v, (jnp.ndarray, np.ndarray)): 
            # likely the most common scenario
            if np.prod(v.shape) == 1: new_v = v.item()
            new_v = v.tolist()  
        elif not isinstance(v, (dict, list)): 
            new_v = repr(v)   # some custom object
        
        new_d[k] = new_v
        
    return new_d
        
############################
# Test parameterization utils via `absl.testing.parametrized`.
############################

def parameters(*testcases):
    """A decorator for creating parameterized tests.

    See the module docstring for a usage example.

    Args:
        *testcases: Parameters for the decorated method, either a single
            iterable, or a list of tuples/dicts/objects (for tests with only one
            argument).

    Raises:
        NoTestsError: Raised when the decorator generates no tests.

    Returns:
        A test generator to be handled by TestGeneratorMetaclass.
    """
    return atp.parameters(*testcases)

def named_parameters(*testcases):
    """A decorator for creating parameterized tests.

    See the module docstring for a usage example. For every parameter tuple
    passed, the first element of the tuple should be a string and will be appended
    to the name of the test method. Each parameter dict passed must have a value
    for the key "testcase_name", the string representation of that value will be
    appended to the name of the test method.

    Args:
        *testcases: Parameters for the decorated method, either a single iterable,
            or a list of tuples or dicts.

    Raises:
        NoTestsError: Raised when the decorator generates no tests.

    Returns:
        A test generator to be handled by TestGeneratorMetaclass.
    """
    return atp.named_parameters(*testcases)

def product(*kwargs_seqs, **testgrid):
    """
    A decorator for running tests over cartesian product of parameters values.

    See the module docstring for a usage example. The test will be run for every
    possible combination of the parameters.

    Args:
        *kwargs_seqs: Each positional parameter is a sequence of keyword arg dicts;
        every test case generated will include exactly one kwargs dict from each
        positional parameter; these will then be merged to form an overall list
        of arguments for the test case.
        **testgrid: A mapping of parameter names and their possible values. Possible
        values should given as either a list or a tuple.

    Raises:
        NoTestsError: Raised when the decorator generates no tests.

    Returns:
        A test generator to be handled by TestGeneratorMetaclass.
    """
    return atp.product(*kwargs_seqs, **testgrid)


############################
# Tensor comparison utils. 
############################

@dataclasses.dataclass
class TestParam:
    r'''
    Class to specify per-tensor relative and absolute tolerances.
    '''
    tensor: jnp.ndarray
    rtol: float
    atol: float


def compare_tensors(
    test,
    reference,
    rtol=1e-5,
    atol=1e-8,
    equal_nan=False,
    msg_prefix='\t',
):
    if test.shape != reference.shape:
        msgs = (
            f'shape mismatch, test: {test.shape}, '
            f'reference: {reference.shape}'
        )
        raise RuntimeError(msgs)

    if test.dtype != reference.dtype:
        msgs = (
            f'dtype mismatch, test: {test.dtype}, '
            f'reference: {reference.dtype}'
        )
        raise RuntimeError(msgs)

    dtype = test.dtype
    input = test.astype(jnp.float32)
    reference = reference.astype(jnp.float32)

    allclose = jnp.allclose(input, reference, rtol, atol, equal_nan)

    abs_diff = jnp.abs(input - reference)

    is_close = abs_diff <= (atol + rtol * jnp.abs(reference))
    close_mask = is_close == True
    not_close_mask = jnp.logical_not(close_mask)
    close_count = close_mask.sum().item()
    total_count = jnp.size(input)
    close_percent = close_count / total_count * 100

    rel_change_denom = jnp.abs(reference)
    rel_change = abs_diff / rel_change_denom
    rel_change.at[rel_change_denom == 0].set(0) # rel_change[rel_change_denom == 0] = 0

    max_mean_change_denom = jnp.maximum(jnp.abs(input), jnp.abs(reference))
    max_mean_change = abs_diff / max_mean_change_denom
    max_mean_change.at[max_mean_change_denom == 0].set(0) # max_mean_change[max_mean_change_denom == 0] = 0

    arith_mean_change_denom = 0.5 * jnp.abs(input + reference)
    arith_mean_change = abs_diff / arith_mean_change_denom
    arith_mean_change.at[arith_mean_change_denom == 0].set(0) # arith_mean_change[arith_mean_change_denom == 0] = 0

    max_abs_diff = abs_diff.max()
    max_rel_change = rel_change.max()
    max_max_mean_change = max_mean_change.max()
    max_arith_mean_change = arith_mean_change.max()

    abs_ref = jnp.abs(reference)
    abs_input = jnp.abs(input)

    msgs = [
        f'allclose: {allclose}',
        f'matched: {close_count} / {total_count} [{close_percent:.2f}%]',
        f'ref range:    {reference.min():11.4e} : {reference.max():11.4e}',
        f'test range:   {input.min():11.4e} : {input.max():11.4e}',
        f'|ref| range:  {abs_ref.min():11.4e} : {abs_ref.max():11.4e}',
        f'|test| range: {abs_input.min():11.4e} : {abs_input.max():11.4e}',
        f'max absolute difference: {max_abs_diff:11.4e}',
        f'max relative change:     {max_rel_change:11.4e}',
        f'max max mean change:     {max_max_mean_change:11.4e}',
        f'max arith mean change:   {max_arith_mean_change:11.4e}',
        f'shape: {input.shape} dtype: {dtype}',  # f'shape: {input.shape} stride: {input.stride()} dtype: {dtype}',
        f'mismatched indices:{not_close_mask.nonzero()}',
    ]
    if msg_prefix is not None:
        msgs = [f'{msg_prefix}{msg}' for msg in msgs]

    return allclose, msgs


############################
# Performance benchmarking utils.
############################

# Temporarily adapted from https://github.com/openai/triton
def benchmark_fn(
    fn,
):
    # Estimate the runtime of the function
    start_time = time.time()
    fn()
    end_time = time.time()
    print(f"Mean initial rep runtime: {(end_time - start_time) * 1000} ms")

    estimate_ms = (end_time - start_time) 
    # compute number of warmup and repeat
    n_warmup = 5 
    n_repeat = 10 

    # Warm-up
    for _ in range(n_warmup):
        fn()
        jax.clear_caches()

    # Benchmark
    n_trials = 100
    times = np.zeros(n_repeat)
    for i in range(n_repeat):
        # record time of `fn` in milliseconds
        cache = cache * 0
        jax.clear_caches()
        fn()
        duration = timeit.timeit(fn, number=n_trials)
        times[i] = duration / n_trials * 1000

    print(f"Mean runtime: {times.mean().item()} ms")
    res = {
        'mean': times.mean().item(),
        'std': times.std().item(),
        'rel_std': (times.std() / times.mean()).item() * 100,
        'median': np.median(times).item(),
        'min': times.min().item(),
        'max': times.max().item(),
        'nrep': len(times),
    }
    return res