import numpy as np
from PIL import Image
import io

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
from FBOutput.FBFrameImage import FBFrameImage
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

    def extract_images_at_frame(self, frame_index, width, height, far):
        """Extract image data for a specific frame."""
        buffer = read_file_from_zip(self.zip_path, f"frame_image{frame_index}.dat")
        if buffer is None:
            return None, None, None

        data_root = FBFrameImage.GetRootAsFBFrameImage(buffer, 0)

        original_rgba = data_root.RgbAsNumpy().tobytes()
        rgba = Image.open(io.BytesIO(original_rgba))
        rgba = np.array(rgba)

        original_depth_data = data_root.DepthAsNumpy()
        depth = np.array(original_depth_data).reshape((height, width))
        depth = np.flipud(depth)
        depth[depth == 0] = far

        original_id_data = data_root.IdAsNumpy()
        segmentation = np.array(original_id_data).reshape((height, width))
        segmentation = np.flipud(segmentation)

        return rgba, depth, segmentation

    def extract_videos(self, Nframe, width, height, far):
        """Extract video data for all frames."""
        rgb = np.empty((Nframe, height, width, 3), dtype=float)
        depth = np.empty([Nframe, height, width], dtype=float)
        segmentation = np.empty([Nframe, height, width], dtype=np.uint32)

        for f in range(Nframe):
            rgb_f, depth_f, seg_f = self.extract_images_at_frame(f, width, height, far)
            rgb[f] = rgb_f[:, :, :3] / 255
            depth[f] = depth_f
            segmentation[f] = seg_f

        return rgb, depth, segmentation

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