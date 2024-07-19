import numpy as np


def extract_vector2_data(vector):
    """Extract 2D vector data from a FlatBuffer vector."""
    return np.array([vector.X(), vector.Y()])


def extract_vector3_data(vector):
    """Extract 3D vector data from a FlatBuffer vector."""
    return np.array([vector.X(), vector.Y(), vector.Z()])


def extract_quaternion_data(quaternion):
    """Extract quaternion data from a FlatBuffer quaternion."""
    return np.array([quaternion.X(), quaternion.Y(), quaternion.Z(), quaternion.W()])
