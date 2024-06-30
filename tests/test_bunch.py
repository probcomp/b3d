import unittest
import jax
import jax.numpy as jnp
import b3d
from b3d.utils import Bunch
import genjax

class MeshTests(unittest.TestCase):
    
    def test_bunch(self):
        b = Bunch(1, 2, 3)
        assert 1 == b[0] and 2 == b[1] and 3 == b[2] 

        b = Bunch(1, x="x", y=3)
        assert 1 == b[0] and "x" == b["x"] and 3 == b["y"] 
        assert 1 == b[0] and "x" == b.x and 3 == b.y

        @genjax.gen
        def model():
            x = genjax.normal(0.,1.) @ "x"
            return Bunch(1, x=x, y=2)

        key = jax.random.PRNGKey(0)
        jsimulate = jax.jit(model.simulate)

        # tr = jsimulate(key, ())
        tr = model.simulate(key, ())
        b = tr.get_retval()

        print(b)

        assert b.x  == tr.get_choices()("x").v, f"{tr.get_choices()('x').v}, {b.x}, {b}"
        assert b[1] == tr.get_choices()("x").v, f"{tr.get_choices()('x').v}, {b[1]}, {b}"