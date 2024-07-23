import importlib

import b3d
import jax
import jax.numpy as jnp

importlib.reload(b3d.renderer.renderer_original)


def test_vmap_rasterize():
    renderer = b3d.RendererOriginal()
    vmap_rasterize = jax.vmap(renderer.rasterize, in_axes=(0, None))

    output = vmap_rasterize(
        jnp.zeros((10, 100, 3)),
        jnp.zeros((10, 3), dtype=jnp.int32),
    )
    assert output.shape == (10, renderer.height, renderer.width, 4)


def test_render():
    renderer = b3d.RendererOriginal()

    output = renderer.render_rgbd(
        jnp.zeros((100, 3)),
        jnp.zeros((10, 3), dtype=jnp.int32),
        jnp.zeros((100, 3)),
    )
    assert output.shape == (renderer.height, renderer.width, 4)


def test_vmap_render():
    renderer = b3d.RendererOriginal()
    vmap_render = jax.vmap(renderer.render_rgbd_many, in_axes=(0, None, None))

    output = vmap_render(
        jnp.zeros((10, 5, 100, 3)),
        jnp.zeros((10, 3), dtype=jnp.int32),
        jnp.zeros((5, 100, 3)),
    )
    assert output.shape == (10, 5, renderer.height, renderer.width, 4)
