import b3d
import pytest


# Arrange
@pytest.fixture
def renderer():
    import b3d
    import pytest

    width = 200
    height = 200
    fx = 200.0
    fy = 200.0
    cx = 100.0
    cy = 100.0
    near = 0.001
    far = 16.0
    renderer = b3d.Renderer(width, height, fx, fy, cx, cy, near, far, 1024)
    return renderer
