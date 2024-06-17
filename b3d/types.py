###
from typing import Any, TypeAlias
import jax
import genjax
from builtins import tuple as _tuple
from jax.tree_util import register_pytree_node_class


Shape = int | tuple[int, ...]
Array: TypeAlias = jax.Array
Bool: TypeAlias = Array
Float: TypeAlias = Array
Int: TypeAlias = Array
Quaternion: TypeAlias = Array
Indexer = int | slice | Array
Matrix = Array
Vector = Array
Direction = Array
GaussianParticle = Any

Key: TypeAlias = jax.Array
Pytree = Any
GenerativeFunction = genjax.GenerativeFunction


@register_pytree_node_class
class NamedArgs:
    def __new__(cls, *args, **kwargs):
        return _tuple.__new__(cls, list(args) + list(kwargs.values()))

    def __init__(self, *args, **kwargs):
        self._d = dict()
        for k, v in kwargs.items():
            self._d[k] = v
            setattr(self, k, v)

    def __getitem__(self, k: str):
        return self._d[k]

    def tree_flatten(self):
        return (self, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
