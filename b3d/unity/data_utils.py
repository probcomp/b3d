import zipfile
import numpy as np

def read_file_from_zip(zip_path, file_name):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        if file_name in zip_ref.namelist():
            with zip_ref.open(file_name) as file:
                file_contents = file.read()
                return bytearray(file_contents)
        else:
            print(f"{file_name} not found in the ZIP archive.")
            return None
        
def extract_vector2_data(vector):
    """Extract 2D vector data from a FlatBuffer vector."""
    return np.array([vector.X(), vector.Y()])

def extract_vector3_data(vector):
    """Extract 3D vector data from a FlatBuffer vector."""
    return np.array([vector.X(), vector.Y(), vector.Z()])

def extract_quaternion_data(quaternion):
    """Extract quaternion data from a FlatBuffer quaternion."""
    return np.array([quaternion.X(), quaternion.Y(), quaternion.Z(), quaternion.W()])
