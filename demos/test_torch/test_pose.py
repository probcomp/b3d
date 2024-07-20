from b3d.renderer.torch.pose import Pose
import torch


def test_as_matrix():
    torch.set_default_device("cuda")
    position = torch.tensor([0.0, 6.3, 0.5])
    quaternion = Pose.unit_quaternion()
    pose = Pose(position, quaternion)
    matrix = pose.as_matrix()
    assert torch.allclose(matrix[:3, 3], position)
    assert torch.allclose(matrix[:3, :3], torch.eye(3))


def test_as_matrix_multi():
    torch.set_default_device("cuda")
    position = torch.tensor([[0.0, 6.3, 0.5], [0.0, 6.3, 0.5]])
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    pose = Pose(position, quaternion)
    pose.as_matrix()
