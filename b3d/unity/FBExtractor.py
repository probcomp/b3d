import numpy as np
from PIL import Image
import io
import OpenEXR
import Imath

from data_utils import read_file_from_zip
from data_utils import extract_vector2_data
from data_utils import extract_vector3_data
from data_utils import extract_quaternion_data

from FBOutput.FBCameraIntrinsics import FBCameraIntrinsics
from FBOutput.FBMetaData import FBMetaData
from FBOutput.FBObjectCatalog import FBObjectCatalog
from FBOutput.FBKeypointsAssignment import FBKeypointsAssignment
from FBOutput.FBObjectPose import FBObjectPose
from FBOutput.FBFrameCameraPose import FBFrameCameraPose
from FBOutput.FBImage import FBImage
from FBOutput.FBColorDict import FBColorDict
from FBOutput.FBFrameKeypoints import FBFrameKeypoints

class FBExtractor:

    def __init__(self, zip_path):
        self.zip_path = zip_path

    def extract_camera_intrinsics(self):
        """Extract camera intrinsics from the ZIP file."""
        buffer = read_file_from_zip(self.zip_path, "camera_intrinsics.dat")
        if buffer is None:
            return None
        
        data_root = FBCameraIntrinsics.GetRootAsFBCameraIntrinsics(buffer, 0)

        width = data_root.Width()
        height = data_root.Height()
        focalLength = data_root.FocalLength()
        sensorSize = extract_vector2_data(data_root.SensorSize())
        lensShift = extract_vector2_data(data_root.LensShift())
        gateFit = data_root.GateFit()
        fov = data_root.Fov()
        near = data_root.NearClipPlane()
        far = data_root.FarClipPlane()

        fx = focalLength * width / sensorSize[0]
        fy = focalLength * height / sensorSize[1]
        cx = width / 2 + width * lensShift[0]
        cy = height / 2 + height * lensShift[1]

        return np.array([width, height, fx, fy, cx, cy, near, far])

    def extract_metadata(self):
        """Extract metadata from the ZIP file."""
        buffer = read_file_from_zip(self.zip_path, "metadata.dat")
        if buffer is None:
            return None
        
        data_root = FBMetaData.GetRootAsFBMetaData(buffer, 0)
        Nframe = data_root.Nframe()
        Nobjects = data_root.Nobjects()
        Nkeypoints = data_root.Nkeypoints()
        samplingrate = data_root.Samplingrate()

        return Nframe, Nobjects, Nkeypoints, samplingrate
    
    
    def extract_file_info(self):
        """Extract file info from the ZIP file and filename."""
        buffer = read_file_from_zip(self.zip_path, "metadata.dat")
        if buffer is None:
            return None
        
        data_root = FBMetaData.GetRootAsFBMetaData(buffer, 0)
        scene_folder = data_root.Scene()
        scene_folder = scene_folder.decode('utf-8').strip("'")

        # from filename
        parts = self.zip_path.split('/')[-1].split('_')
            
        if len(parts) < 4:
            raise ValueError("Filename does not have the expected format.")
    
        base_name = parts[0]
        light_setting = parts[1]
        background_setting = parts[2]
        resolution = parts[3].split('.')[0]

        return {
            'scene_folder': scene_folder,
            'base_name': base_name,
            'light_setting': light_setting,
            'background_setting': background_setting,
            'resolution': resolution
        }

    def extract_object_catalog(self):
        """Extract object catalog from the ZIP file."""
        buffer = read_file_from_zip(self.zip_path, "object_catalog.dat")
        if buffer is None:
            return None
        
        data_root = FBObjectCatalog.GetRootAsFBObjectCatalog(buffer, 0)

        return [data_root.ObjectCatalog(i).decode('utf-8') for i in range(data_root.ObjectCatalogLength())]

    def extract_keypoints_object_assignment(self):
        """Extract keypoints object assignment from the ZIP file."""
        buffer = read_file_from_zip(self.zip_path, "keypoints_object_assignment.dat")
        if buffer is None:
            return None
        
        data_root = FBKeypointsAssignment.GetRootAsFBKeypointsAssignment(buffer, 0)
        return data_root.ObjectAssignmentsAsNumpy()

    def extract_object_poses_from_file(self, filename):
        """Extract object poses from a specific file in the ZIP archive."""
        buffer = read_file_from_zip(self.zip_path, filename)
        if buffer is None:
            return None, None
        
        data_root = FBObjectPose.GetRootAsFBObjectPose(buffer, 0)

        positions = np.array([extract_vector3_data(data_root.Positions(i)) for i in range(data_root.PositionsLength())])
        quaternions = np.array([extract_quaternion_data(data_root.Quaternions(i)) for i in range(data_root.QuaternionsLength())])

        return positions, quaternions

    def extract_object_poses(self, Nframe):
        """Extract object poses for all frames from the ZIP file."""
        positions = []
        quaternions = []

        staticObjectsPosition, staticObjectsQuaternion = self.extract_object_poses_from_file("static_objects.dat")

        for f in range(Nframe):
            dyn_position, dyn_quaternion = self.extract_object_poses_from_file(f"frame_objects{f}.dat")
            position = np.concatenate([dyn_position, staticObjectsPosition], axis=0)
            quaternion = np.concatenate([dyn_quaternion, staticObjectsQuaternion], axis=0)
            positions.append(position)
            quaternions.append(quaternion)

        return np.array(positions), np.array(quaternions)

    def extract_camera_pose_at_frame(self, frame_index):
        """Extract camera pose at a specific frame index."""
        buffer = read_file_from_zip(self.zip_path, f"frame_camera_pose{frame_index}.dat")
        if buffer is None:
            return None, None
        
        data_root = FBFrameCameraPose.GetRootAsFBFrameCameraPose(buffer, 0)

        position = extract_vector3_data(data_root.Position())
        quaternion = extract_quaternion_data(data_root.Quaternion())

        return position, quaternion

    def extract_camera_poses(self, Nframe):
        """Extract camera poses for all frames."""
        camera_position = np.empty((Nframe, 3), dtype=float)
        camera_rotation = np.empty((Nframe, 4), dtype=float)
        for f in range(Nframe):
            position, quaternion = self.extract_camera_pose_at_frame(f)
            camera_position[f] = position
            camera_rotation[f] = quaternion

        return camera_position, camera_rotation

    def extract_png_image_at_frame(self, frame_index, image_pass='rgb'):
        """Extract image data for a specific frame. image_pass can take values 'rgb' or 'seg'."""
        buffer = read_file_from_zip(self.zip_path, f"frame_{image_pass}{frame_index}.dat")
        if buffer is None:
            return None

        data_root = FBImage.GetRootAsFBImage(buffer, 0)

        raw_img = data_root.ImageAsNumpy().tobytes()
        img = Image.open(io.BytesIO(raw_img))
        img = np.array(img)

        return img
    
    def extract_colordict(self):
        """Extract segmentation color - object id dictionary."""
        buffer = read_file_from_zip(self.zip_path, "color_objectid_dict.dat")
        if buffer is None:
            return None
        
        data_root = FBColorDict.GetRootAsFBColorDict(buffer, 0)

        colordict = {}
        num_entries = data_root.EntriesLength()
        
        for i in range(num_entries):
            entry = data_root.Entries(i)
            key = entry.Key()
            value = entry.Value()
            color = (key.R(), key.G(), key.B(), key.A())
            colordict[color] = value

        # add segmentation color 'white', which corresponds to the skybox/empty. Matching it to uintmax=4294967295
        white = (255, 255, 255, 255)
        colordict[white] = 4294967295

        return colordict
    
    def extract_segmentation_at_frame(self, frame_index, width, height, color_to_int):
        """Extract segmentation image."""
        seg_img = self.extract_png_image_at_frame(frame_index, 'seg')

        flattened_image = seg_img.reshape(-1, 4)

        seg = np.array([color_to_int.get(tuple(pixel), 0) for pixel in flattened_image], dtype=np.uint)
        seg = seg.reshape(height, width)

        return seg

    def extract_depth_at_frame(self, frame_index, width, height, far, CHANNELS = ['R', 'G', 'B', 'A']):
        """Extract depth data for a specific frame. """
        buffer = read_file_from_zip(self.zip_path, f"frame_depth{frame_index}.dat")
        if buffer is None:
            return None

        data_root = FBImage.GetRootAsFBImage(buffer, 0)
        
        raw_depth = data_root.ImageAsNumpy().tobytes()

        # Depth values were encoded as EXR (HDR)
        exr_stream = io.BytesIO(raw_depth)
        exr_file = OpenEXR.InputFile(exr_stream)
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        
        # Read the channels
        channel_data = {
            c: np.frombuffer(exr_file.channel(c, pt), dtype=np.float32).reshape((height, width))
            for c in CHANNELS
        }

        # Depth values are stored in the R channel
        depth_c = channel_data['R']

        # Value corresponding to 0 corresponds to the skybox/empty set these values to the far plane
        depth = np.where(depth_c == 0, far, depth_c)

        return depth
    
    def extract_rgb(self, Nframe, width, height):
        """Extract rgb data for all frames."""
        rgb = np.empty((Nframe, height, width, 3), dtype=float)

        for f in range(Nframe):
            rgb_f = self.extract_png_image_at_frame(f, 'rgb')
            rgb[f] = rgb_f[:, :, :3] / 255 # as float

        return rgb
    
    def extract_segmentation(self, Nframe, width, height):
        """Extract rgb data for all frames."""
        segmentation = np.empty((Nframe, height, width), dtype=np.uint32)

        colordict = self.extract_colordict()
        color_to_int = {tuple(color): value for color, value in colordict.items()}
        for f in range(Nframe):
            segmentation[f] = self.extract_segmentation_at_frame(f, width, height, color_to_int)

        return segmentation
    
    def extract_depth(self, Nframe, width, height, far):
        """Extract rgb data for all frames."""
        depth = np.empty((Nframe, height, width), dtype=float)

        CHANNELS = ['R', 'G', 'B', 'A']
        for f in range(Nframe):
            depth[f] = self.extract_depth_at_frame(f, width, height, far, CHANNELS)

        return depth

    def extract_keypoints_data_at_framet(self, frame_index):
        """Extract keypoints data at a specific frame index."""
        buffer = read_file_from_zip(self.zip_path, f"frame_kp{frame_index}.dat")
        if buffer is None:
            return None, None

        data_root = FBFrameKeypoints.GetRootAsFBFrameKeypoints(buffer, 0)

        positions = np.array([extract_vector3_data(data_root.Positions(i)) for i in range(data_root.PositionsLength())])
        visibilities = data_root.VisibilitiesAsNumpy()

        return positions, visibilities

    def extract_keypoints(self, Nframe, Nkeypoints):
        """Extract keypoints data for all frames."""
        keypoint_positions = np.empty((Nframe, Nkeypoints, 3), dtype=float)
        keypoint_visibility = np.empty((Nframe, Nkeypoints), dtype=bool)

        for f in range(Nframe):
            k_position, k_visibility = self.extract_keypoints_data_at_framet(f)
            keypoint_positions[f] = k_position
            keypoint_visibility[f] = k_visibility

        return keypoint_positions, keypoint_visibility