import b3d
import pytest

# Arrange
@pytest.fixture
def renderer():
    width=100
    height=100
    fx=50.0
    fy=50.0
    cx=50.0
    cy=50.0
    near=0.001
    far=16.0
    return b3d.Renderer(
        width, height, fx, fy, cx, cy, near, far
    )
