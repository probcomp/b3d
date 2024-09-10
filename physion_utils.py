# import io
# import os

# import h5py
# import matplotlib.pyplot as plt
# import numpy as np
# import trimesh
# from PIL import Image

# # Serialize a set of mask objects given by the data loader
# # from save_feats import device


# def get_object_masks(seg_imgs, seg_colors, background=True):
#     # if len(seg_imgs.shape) == 3:
#     #     seg_imgs = np.expand_dims(seg_imgs, 0)
#     #     is_batch = False
#     # else:
#     #     is_batch = True

#     obj_masks = []
#     for scol in seg_colors:
#         mask = seg_imgs == scol  # .astype(np.float)

#         # If object is not visible in the frame
#         # if mask.sum() == 0:
#         #     mask[:] = 1
#         obj_masks.append(mask)

#     obj_masks = np.stack(obj_masks, 0)
#     obj_masks = obj_masks.min(axis=-1)  # collapse last channel dim

#     if background:
#         bg_mask = ~obj_masks.max(0, keepdims=True)
#         obj_masks = np.concatenate([obj_masks, bg_mask], 0)
#     obj_masks = np.expand_dims(obj_masks, -3)  # make N x 1 x H x W

#     # if not is_batch:
#     #     obj_masks = obj_masks[0]
#     return obj_masks


# def get_target_area(file_path):
#     with h5py.File(file_path, "r") as f:
#         rgb_img, seg_img = index_img(f, 0, "_cam0")

#         target_seg_color = f["static"]["object_segmentation_colors"][0]

#         target_seg_color = target_seg_color[None, None, :]

#         arr = seg_img == target_seg_color

#         arr = arr.min(-1).astype("float")

#         area = arr.sum() / (arr.shape[0] ** 2)
#     return file_path, area


# from abc import ABC, abstractmethod


# class Encoding(ABC):
#     def __init__(self, config):
#         self.config = config
#         self.encoding = None

#     # TODO: can add methods to normalize the encodings and unnormalize them. But not sure if they should go here. they are some global property of the dataset.
#     # in future we can simply make this an nn.module which would enable training the entire system end to end.
#     @abstractmethod
#     def encode_shape_representation(self, h5_file, scenario):
#         """
#         :param h5_file: this contains a video with the raw mesh + rigid transformations saved from TDW.
#         :return: encoding: seq_len x nobj x encoding size
#         also sets self.encoding = encoding
#         """

#     @abstractmethod
#     def draw_encoding(self, ax, colors, encoding):  # decoding basically
#         """
#         :param ax: matplotlib axis object on which to draw stuff
#         :param colors: list of colors [nobj, 3]
#         :param encoding: seq_len x nobj x encoding size.
#         :return: updated axis object
#         """


# class BBoxTriMeshEncoding(Encoding):
#     def __init__(self, config, mean=0, std=1):
#         super().__init__(config)
#         self.mean = mean
#         self.std = std
#         self.pts = torch.tensor(
#             [
#                 [-1, -1, -1],
#                 [-1, -1, 1],
#                 [-1, 1, 1],
#                 [-1, 1, -1],
#                 [1, -1, -1],
#                 [1, -1, 1],
#                 [1, 1, 1],
#                 [1, 1, -1],
#             ]
#         )
#         self.all_edges = [
#             (0, 1),
#             (1, 2),
#             (2, 3),
#             (3, 0),
#             (4, 5),
#             (5, 6),
#             (6, 7),
#             (7, 4),
#             (0, 4),
#             (1, 5),
#             (2, 6),
#             (3, 7),
#         ]

#         self.all_faces = [
#             (0, 1, 2, 3),
#             (4, 5, 6, 7),
#             (2, 3, 6, 7),
#             (1, 0, 4, 5),
#             (1, 2, 6, 5),
#             (0, 4, 7, 3),
#         ]

#     # def encode_shape_representation(self, img): where img is a torch tensor image

#     # TODO: include other fns for normalizing repn, loss functions applicable here etc.
#     def encode_shape_representation(self, h5_file, scenario, return_mesh=False):
#         """
#         :param h5_file: this contains a video with the raw mesh + rigid transformations saved from TDW.
#         :return: encoding: seq_len x nobj x encoding size
#         also sets self.encoding = encoding
#         labels: True or False. i.e. contact/no_contact
#         """

#         agent = np.array(h5_file["static"]["target_id"]).item()
#         patient = np.array(h5_file["static"]["zone_id"]).item()

#         colors = []

#         object_ids = np.array(h5_file["static"]["object_ids"])

#         object_id_to_ind = {}
#         for ct, obj_id in enumerate(object_ids):
#             object_id_to_ind[obj_id] = ct

#         colors += list(getDistinctColors(len(object_ids)))

#         colors[object_id_to_ind[patient]] = [0, 1, 0]
#         colors[object_id_to_ind[agent]] = [1, 0, 0]

#         self.colors = colors

#         n_frames = len(h5_file["frames"].keys())

#         indices = np.arange(n_frames)

#         if not return_mesh:
#             encoding = get_bbox_encoding_seq(
#                 h5_file, scenario, indices, save_mass=True, save_mesh=True
#             )  # seq len, nobj, 9
#             self.encoding = encoding
#             return encoding
#         else:
#             encoding, mesh_seq = get_bbox_encoding_seq(
#                 h5_file,
#                 scenario,
#                 indices,
#                 save_mass=True,
#                 save_mesh=True,
#                 return_mesh=True,
#             )  # seq len, nobj, 9
#             self.encoding = encoding
#             return encoding, mesh_seq

#     def check_collision(self, m1, m2):
#         #     request = fcl.CollisionRequest()
#         #     result = fcl.CollisionResult()

#         request = fcl.DistanceRequest(enable_signed_distance=True)
#         result = fcl.DistanceResult()

#         ret = fcl.distance(m1, m2, request, result)

#         return ret

#     def get_collision_obj(self, mesh):
#         """
#         :param repn: [feat_size] array for bbox repn
#         :return: BVH collision object using fcl library
#         """

#         #     all_pts = data_utils.decode_numpy(repn[:3], repn[3:6], repn[6:9], self.pts.numpy())
#         # mesh = trimesh.Trimesh(vertices=all_pts,
#         #                        faces=all_faces)

#         verts = mesh.vertices
#         tris = mesh.faces

#         m = fcl.BVHModel()
#         m.beginModel(len(verts), len(tris))
#         m.addSubModel(verts, tris)
#         m.endModel()

#         m_coll = fcl.CollisionObject(m)

#         return m_coll

#     def check_mesh_coll(self, m1, m2):
#         m1 = self.get_collision_obj(m1)
#         m2 = self.get_collision_obj(m2)
#         return self.check_collision(m1, m2)

#     def get_scale(self, pts):
#         size_raw = np.max(pts, 0) - np.min(pts, 0)
#         # size_raw /= 2
#         size_raw = np.clip(size_raw, -2, 2)
#         return size_raw

#     def draw_encoding(self, ax, colors, encoding):  # decoding basically
#         """
#         :param ax: matplotlib axis object on which to draw stuff
#         :param colors: list of colors [nobj, 3]
#         :param encoding: nobj x encoding size.
#         :return: updated axis object
#         """
#         if encoding is None:
#             encoding = self.encoding

#         for ct, trimesh_mesh in enumerate(encoding):
#             v = trimesh_mesh.vertices  # [:, [0, 2, 1]]

#             f = trimesh_mesh.faces

#             pc = art3d.Poly3DCollection(
#                 v[f], facecolors=colors[ct], edgecolor="black", alpha=0.5
#             )
#             ax.add_collection(pc)

#         return ax

#     def get_max_interpenetration(self, mesh_seq):
#         max_volume = -10

#         for seq in mesh_seq:
#             for ct, pair in enumerate(list(itertools.combinations(seq, 2))):
#                 m1, m2 = pair
#                 coll_dist = self.check_mesh_coll(m1, m2)
#                 if coll_dist == 0:
#                     intersect = trimesh.boolean.intersection([m1, m2], engine="blender")
#                     if not intersect.is_empty and intersect.volume > max_volume:
#                         max_volume = intersect.volume

#         return max_volume

#     def check_floor_penetration(self, mesh_seq):
#         penetrated = 1000

#         for mesh_list in mesh_seq:
#             for mesh in mesh_list:
#                 dist = mesh.vertices[:, 1].min().item()
#                 if dist < penetrated:
#                     penetrated = dist

#         return penetrated

#     def decode_bbox(self, encoding):
#         """
#         :param data:
#             encoding: a list of [seq_len, n_objs, feat_size] tensors -> n_objs can be variable
#         :return:
#             decoding: a list of [seq_len, n_objs, 8, 3] tensors -> n_objs can be variable
#         """

#         all_pts = decode_bbox_torch(encoding)
#         # list of [seq, nobj, 8, 3]

#         return all_pts


# def get_penetration(file_path):
#     with h5py.File(file_path) as h5file:
#         obj = BBoxTriMeshEncoding(None)

#         encoding, mesh_seq = obj.encode_shape_representation(
#             h5file, "link", return_mesh=True
#         )

#         floor_penetration = obj.check_floor_penetration(mesh_seq)

#         #         object_penetration = obj.get_max_interpenetration(mesh_seq)

#         #         file = open(file_path.split('.')[0] + '_floor_penet.txt', 'w')
#         #         file.write(str(floor_penetration))
#         #         file.close()

#         #         file = open(all_files[0].split('.')[0] + '_obj_penet.txt', 'w')
#         #         file.write(str(object_penetration))
#         #         file.close()

#         return file_path, floor_penetration, floor_penetration


# # pilot_linking_nl1-8_mg000_aCyl_bCyl_tdwroom1
# def accept_stimuli(filepath, check_interp=True, check_area=True):
#     # breakpoint()
#     if (not check_interp) and (not check_area):
#         return True

#     # breakpoint()
#     area = get_target_area(filepath)[1]

#     if check_area:
#         if "small_zone" in filepath:
#             if area < 0.0002:  # i.e out of the frame
#                 return False
#             else:
#                 return True
#         elif "dominoes" in filepath:
#             if area < 0.015:
#                 return False
#             else:
#                 return True
#         elif area < 0.02:
#             return False
#         elif "yeet" in filepath:
#             return True
#     if check_interp and (get_penetration(filepath)[1] < -0.02):
#         print("Penetration too high")
#         return False
#     return True


# def concat_masks(objects_masked, mask=False):
#     """
#     objects_masked: list of {1 x n_objects x seq_len x img_size x img_size x 3} tensors
#     returns: concatenated list of objects_masked {n_objects*seq_len*batch_size x img_size x img_size x 3}
#     """
#     all_objects = []
#     img_size = 256  # objects_masked[0].shape[-2]
#     for xx in objects_masked:
#         if not mask:
#             xx = xx.view(-1, img_size, img_size, 3)
#         else:
#             xx = xx.view(-1, img_size, img_size)
#         #         print(xx.shape)
#         all_objects.append(xx)
#     all_objects = torch.cat(all_objects, 0)
#     return all_objects


# # Get an image from a raw img tensor taken from a tdw saved video
# def get_image(raw_img, pil=False):
#     """
#     raw_img: binary image to be read by PIL
#     returns: HxWx3 image
#     """
#     img = Image.open(io.BytesIO(raw_img))

#     if pil:
#         return img

#     return np.array(img)


# # apply object masks to source rgb image to get cropped images
# def apply_masks(all_masks, rgb_img):
#     """
#     all_masks: n_objectsxHxWx3 binary mask images, one per object in rgb_img
#     rgb_img: source image HxWx3
#     """
#     n_objects = len(all_masks)
#     rgb_img = np.stack([rgb_img] * n_objects, 0)
#     #     print(rgb_img.shape)
#     masked_objects = rgb_img * np.expand_dims(all_masks, -1)
#     return masked_objects


# def get_num_frames(h5_file):
#     return len(h5_file["frames"].keys())


# def index_imgs(h5_file, indices, static=False, suffix="", pil=False):
#     if not static:
#         all_imgs = []
#         all_segs = []
#         for ct, index in enumerate(indices):
#             rgb_img, segments = index_img(h5_file, index, suffix=suffix, pil=pil)
#             all_imgs.append(rgb_img)
#             all_segs.append(segments)
#     else:
#         rgb_img, segments = index_img(h5_file, indices[0])
#         all_imgs = [rgb_img]
#         all_segs = [segments]

#         for ct, index in enumerate(indices[1:]):
#             all_imgs.append(rgb_img)
#             all_segs.append(segments)

#     if not pil:
#         all_imgs = np.stack(all_imgs, 0)
#     all_segs = np.stack(all_segs, 0)

#     return all_imgs, all_segs


# def get_video(h5_file):
#     all_imgs = []
#     indices = np.arange(0, len(h5_file["frames"]))
#     for ct, index in enumerate(indices):
#         rgb_img, _ = index_img(h5_file, index, suffix="_cam0")
#         all_imgs.append(rgb_img)
#     return all_imgs


# def plot_video(h5_file_path):
#     with h5py.File(h5_file_path) as h5_file:
#         vid = get_video(h5_file)
#         vid = np.stack(vid)
#         plot_sequence_images(vid)


# def index_img(h5_file, index, suffix="", pil=False):
#     # print(index)

#     img0 = h5_file["frames"][str(index).zfill(4)]["images"]

#     rgb_img = get_image(img0["_img" + suffix][:], pil=pil)

#     segments = np.array(get_image(img0["_id" + suffix][:]))

#     return rgb_img, segments


# def visualize(all_masks, all_objects, rgb_img):
#     summed_object = all_objects.sum(0)

#     n_objects = len(all_objects)

#     fig = plt.figure(figsize=[12, n_objects * 4])

#     n_images = n_objects * 2

#     for xx in range(n_objects):
#         ax = fig.add_subplot(n_objects, 3, xx * 3 + 1)
#         ax.set_axis_off()
#         ax.imshow(all_masks[xx])

#         ax = fig.add_subplot(n_objects, 3, xx * 3 + 2)
#         ax.set_axis_off()
#         ax.imshow(all_objects[xx])

#     ax = fig.add_subplot(n_objects, 3, 3)
#     ax.set_axis_off()
#     ax.imshow(summed_object)

#     ax = fig.add_subplot(n_objects, 3, 6)
#     ax.set_axis_off()
#     ax.imshow(rgb_img)

#     fig.savefig("./test_dl.png")

#     plt.close(fig)


# def get_ax(fig):
#     ax = fig.add_subplot(projection="3d")
#     ax.set_xlim(-2, 2)
#     ax.set_ylim(-0.5, 2)
#     ax.set_zlim(-2, 2)
#     ax.view_init(elev=30.0, azim=30, vertical_axis="y")
#     return ax


# def plot_box(pts, colors, close=False):
#     fig = plt.figure(figsize=[20, 20])
#     ax = fig.add_subplot(projection="3d")
#     ax.set_xlim(-2, 2)
#     ax.set_ylim(-2, 2)
#     ax.set_zlim(-2, 2)
#     ax.view_init(elev=100.0, azim=-90)
#     ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors)
#     return fig


# #     plt.close(fig)
# # plot_box(vertices_orig, colors=[[0, 0, 1]])


# def get_transformed_pts(pts, rotations, trans):
#     #     frame = str(frame).zfill(4)

#     rot = R.from_quat(rotations).as_matrix()
#     transformed_pts = np.matmul(rot, pts.T).T + np.expand_dims(trans, axis=0)

#     return transformed_pts


# import colorsys


# def HSVToRGB(h, s, v):
#     (r, g, b) = colorsys.hsv_to_rgb(h, s, v)
#     return (int(255 * r), int(255 * g), int(255 * b))


# def getDistinctColors(n):
#     huePartition = 1.0 / (n + 1)
#     return (
#         np.array(
#             list((HSVToRGB(huePartition * value, 0.4, 0.5) for value in range(0, n)))
#         )
#         / 255.0
#     )


# def get_vertices_scaled(f, obj_id, ob):
#     vertices_orig_unscaled = np.array(f["static"]["mesh"]["vertices_" + str(ob)])

#     vertices_orig = np.copy(vertices_orig_unscaled)

#     scales = f["static"]["scale"][:]

#     # scales1 = copy.deepcopy(scales[1])
#     # scales[1] = copy.deepcopy(scales[2])
#     # scales[2] = scales1

#     vertices_orig[:, 0] *= scales[ob, 0]
#     vertices_orig[:, 1] *= scales[ob, 1]
#     vertices_orig[:, 2] *= scales[ob, 2]
#     faces_orig = np.array(f["static"]["mesh"]["faces_" + str(ob)])

#     return vertices_orig, vertices_orig_unscaled, faces_orig


# def draw3DRectangle(ax, x1, y1, z1, x2, y2, z2, color):
#     #     ax.set_xlim(-2, 2)
#     #     ax.set_ylim(-2, 2)
#     #     ax.set_zlim(-2, 2)
#     # the Translate the datatwo sets of coordinates form the apposite diagonal points of a cuboid
#     ax.plot([x1, x2], [y1, y1], [z1, z1], color=color)  # | (up)
#     ax.plot([x2, x2], [y1, y2], [z1, z1], color=color)  # -->
#     ax.plot([x2, x1], [y2, y2], [z1, z1], color=color)  # | (down)
#     ax.plot([x1, x1], [y2, y1], [z1, z1], color=color)  # <--

#     ax.plot([x1, x2], [y1, y1], [z2, z2], color=color)  # | (up)
#     ax.plot([x2, x2], [y1, y2], [z2, z2], color=color)  # -->
#     ax.plot([x2, x1], [y2, y2], [z2, z2], color=color)  # | (down)
#     ax.plot([x1, x1], [y2, y1], [z2, z2], color=color)  # <--

#     ax.plot([x1, x1], [y1, y1], [z1, z2], color=color)  # | (up)
#     ax.plot([x2, x2], [y2, y2], [z1, z2], color=color)  # -->
#     ax.plot([x1, x1], [y2, y2], [z1, z2], color=color)  # | (down)
#     ax.plot([x2, x2], [y1, y1], [z1, z2], color=color)  # <--


# def draw3DRectangleBox(ax, edges, pts, color):
#     #     ax.set_xlim(-2, 2)
#     #     ax.set_ylim(-2, 2)
#     #     ax.set_zlim(-2, 2)
#     # the Translate the data two sets of coordinates form the apposite diagonal points of a cuboid
#     for edge in edges:
#         x1, y1, z1 = pts[edge[0]]
#         x2, y2, z2 = pts[edge[1]]
#         ax.plot([x1, x2], [y1, y2], [z1, z2], color=color)  # | (up

#     ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color)


# def get_full_bbox(vertices):
#     arr1 = vertices.min(0)

#     arr2 = vertices.max(0)

#     arr = np.stack([arr1, arr2], 0)

#     pts = [
#         (0, 0, 0),
#         (0, 0, 1),
#         (0, 1, 1),
#         (0, 1, 0),
#         (1, 0, 0),
#         (1, 0, 1),
#         (1, 1, 1),
#         (1, 1, 0),
#     ]

#     all_edges = [
#         (0, 1),
#         (1, 2),
#         (2, 3),
#         (3, 0),
#         (4, 5),
#         (5, 6),
#         (6, 7),
#         (7, 4),
#         (0, 4),
#         (1, 5),
#         (2, 6),
#         (3, 7),
#     ]

#     all_faces = [
#         (0, 1, 2, 3),
#         (4, 5, 6, 7),
#         (2, 3, 6, 7),
#         (1, 0, 4, 5),
#         (1, 2, 6, 5),
#         (0, 4, 7, 3),
#     ]

#     index = np.arange(3)

#     all_pts = []
#     for pt in pts:
#         p1 = arr[pt, index]
#         all_pts.append(p1)

#     all_pts = np.stack(all_pts, 0)

#     return all_pts, all_edges, all_faces


# def scatter_pts(pts):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")
#     ax.set_xlim(-2, 2)
#     ax.set_ylim(-2, 2)
#     ax.set_zlim(-2, 2)
#     ax.view_init(elev=100.0, azim=-90)
#     ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
#     return


# import torch
# from sympy import *


# def asCartesian(rthetaphi):
#     # takes list rthetaphi (single coord)
#     r = rthetaphi[0]
#     theta = rthetaphi[1] * pi / 180  # to radian
#     phi = rthetaphi[2] * pi / 180
#     x = r * sin(theta) * cos(phi)
#     y = r * sin(theta) * sin(phi)
#     z = r * cos(theta)
#     return [x, y, z]


# def asSpherical(xyz):
#     # takes list xyz (single coord)
#     x = xyz[0]
#     y = xyz[1]
#     z = xyz[2]
#     r = sqrt(x * x + y * y + z * z)
#     theta = acos(z / r) * 180 / pi  # to degrees
#     phi = atan2(y, x) * 180 / pi
#     return [r, theta, phi]


# from scipy.spatial.transform import Rotation as R


# def norm(vector):
#     return vector / np.sqrt(np.sum(vector**2) + 1e-14)


# def get_rotation(pts):
#     import pytorch3d
#     import pytorch3d.transforms

#     x_axis = norm(pts[4] - pts[0])
#     y_axis = norm(pts[2] - pts[1])
#     z_axis = norm(pts[1] - pts[0])

#     rotation_matrix = np.stack([x_axis, y_axis, z_axis], 0)

#     euler_angles = pytorch3d.transforms.matrix_to_euler_angles(
#         torch.tensor(rotation_matrix), "XYZ"
#     )

#     #     eul.mat2euler(rotation_matrix, axes='sxyz')

#     mat = pytorch3d.transforms.euler_angles_to_matrix(euler_angles, "XYZ").numpy()

#     #     mat = eul.euler2mat(euler_angles[0], euler_angles[1], euler_angles[2], axes='sxyz')

#     assert abs(np.sum(np.matmul(rotation_matrix, rotation_matrix.T)) - 3) < 1e-3

#     #     print("diff", np.mean((mat - rotation_matrix)**2))

#     return euler_angles.numpy()


# def decode_numpy(centroid, euler_angles, size, pts):
#     """

#     :param centroid: 3,
#     :param euler_angles: 3,
#     :param size: 3,
#     :param pts: [N, 3] np array
#     :return:
#     """

#     mat = pytorch3d.transforms.euler_angles_to_matrix(
#         torch.tensor(euler_angles), "XYZ"
#     ).numpy()
#     # print(euler_angles)

#     all_pts = []
#     for pt in pts:
#         p1 = centroid + np.matmul(mat.T, size * pt)
#         all_pts.append(p1)

#     all_pts = np.stack(all_pts, 0)

#     return all_pts


# import cv2


# def write_video(file_path, frames, fps):
#     """
#     Writes frames to an mp4 video file
#     :param file_path: Path to output video, must end with .mp4
#     :param frames: List of PIL.Image objects
#     :param fps: Desired frame rate
#     """

#     h, w = frames[0].shape[:2]
#     fourcc = cv2.cv2.VideoWriter_fourcc(
#         *"MJPG"
#     )  # cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#     writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

#     for frame in frames:
#         writer.write((frame[:, :, [2, 1, 0]] * 255).astype("uint8"))

#     writer.release()


# def plot_sequence_images(image_array, fps=100):
#     """Display images sequence as an animation in jupyter notebook

#     Args:
#         image_array(numpy.ndarray): image_array.shape equal to (num_images, height, width, num_channels)
#     """
#     from IPython.display import HTML, display
#     from matplotlib import animation

#     dpi = 72.0
#     xpixels, ypixels = image_array[0].shape[:2]
#     fig = plt.figure(figsize=(ypixels / dpi, xpixels / dpi), dpi=dpi)
#     im = plt.figimage(image_array[0])

#     def animate(i):
#         im.set_array(image_array[i])
#         return (im,)

#     anim = animation.FuncAnimation(
#         fig,
#         animate,
#         frames=len(image_array),
#         interval=1000 / fps,
#         repeat_delay=1,
#         repeat=True,
#     )
#     plt.show()
#     display(HTML(anim.to_html5_video()))


# def decode_and_draw(
#     encoding,
#     seed_len,
#     encoding_object,
#     exp_path,
#     label="None",
#     colors=None,
#     fig_title="test",
#     figsize=10,
#     loss="0.0",
# ):
#     tmp_file_path = os.path.join(exp_path, "temp.png")

#     if colors is None:
#         colors = [[0, 1, 0], [1, 0, 0]]

#         colors += list(getDistinctColors(encoding.shape[1] - 2))

#     all_imgs = []

#     for ct_, enc in enumerate(encoding):
#         # enc: n_obj x 9
#         fig = plt.figure(figsize=[figsize, figsize])

#         ax = get_ax(fig)

#         ax = encoding_object.draw_encoding(ax, colors, enc)

#         if ct_ < seed_len:
#             ax.set_title("Input (GT)", fontsize=16, color="b")
#         else:
#             ax.set_title("Simulated (" + str(fig_title) + ")", fontsize=16, color="r")

#         fig.text(0.45, 0.25, "OCP: " + str(label), size=12, fontweight="bold")
#         fig.text(0.45, 0.20, "L:" + loss, size=12, fontweight="bold")

#         fig.savefig(tmp_file_path)

#         plt.close(fig)

#         img = plt.imread(tmp_file_path)

#         all_imgs.append(img.transpose([2, 0, 1])[:4])

#     # os.system('rm -rf ' +  tmp_file_path)

#     all_imgs = np.expand_dims(np.stack(all_imgs, 0), 0)

#     return all_imgs


# def rotate_and_translate(positions, euler_angles, size, pts):
#     """

#     :param euler_angles: [N, 3]
#     :param positions: [N, 3]
#     :param size: [N, 3]
#     :param pts: [N, W, 3]
#     :return: all_pts: transformed pts of size [N, W, 3]
#     """
#     import pytorch3d
#     import pytorch3d.transforms

#     mat = pytorch3d.transforms.euler_angles_to_matrix(euler_angles, "XYZ").permute(
#         0, 2, 1
#     )  # [seq*totalnobj, 3, 3]

#     scaled_pts = size.unsqueeze(1) * pts  # [seq*totalnobj, 8, 3]
#     scaled_pts = scaled_pts.permute(0, 2, 1)

#     rotated_pts = torch.matmul(mat, scaled_pts)  # [seq*totalnobj, 3, 8]
#     rotated_pts = rotated_pts.permute(0, 2, 1)  # [seq*totalnobj, 8, 3]

#     all_pts = positions.unsqueeze(1) + rotated_pts  # [seq*totalnobj, 8, 3]

#     all_pts = all_pts.contiguous()

#     return all_pts

#     # [seq, totalnobj, 8, 3]


# def decode_bbox_torch(encoding):
#     """
#     :param data:
#         encoding: a list of [seq_len, n_objs, feat_size] tensors -> n_objs can be variable
#     :return:
#         decoding: a list of [seq_len, n_objs, 8, 3] tensors -> n_objs can be variable
#     """

#     # print(euler_angles.shape)

#     n_objs = get_n_objs_from_encoding(encoding)

#     all_objs = torch.cat(encoding, 1)  # [seq_len, total_nobjs, feat_size]
#     seq_len = all_objs.shape[0]
#     total_obj = all_objs.shape[1]

#     all_objs = all_objs.view(-1, all_objs.shape[-1])

#     positions = all_objs[:, :3]
#     euler_angles = all_objs[:, 3:6]
#     size = all_objs[:, 6:9]  # [seq_len * total_nobjs, 3]

#     pts = torch.tensor(
#         [
#             [-1, -1, -1],
#             [-1, -1, 1],
#             [-1, 1, 1],
#             [-1, 1, -1],
#             [1, -1, -1],
#             [1, -1, 1],
#             [1, 1, 1],
#             [1, 1, -1],
#         ]
#     )

#     pts = torch.stack([pts] * positions.shape[0], 0).to(
#         positions.device
#     )  # [seq*totalnobj, 8, 3]

#     all_pts = rotate_and_translate(positions, euler_angles, size, pts)

#     all_pts = all_pts.view(seq_len, total_obj, all_pts.shape[-1], all_pts.shape[-2])

#     all_pts = unpack_fullnobj_tensor(all_pts, n_objs)

#     return all_pts


# def get_bbox_encoding_seq(
#     f,
#     scenario,
#     indices,
#     save_mass=False,
#     save_mesh=False,
#     prefix="_cam0",
#     return_mesh=False,
# ):
#     # reject = np.array(f['static']['model_names']) != b'cube'

#     m_names = np.array(f["static"]["model_names"])
#     occluders = np.array(f["static"]["occluders"])
#     distractors = np.array(f["static"]["distractors"])

#     mass = np.array(f["static"]["mass"])

#     # print("m_names", m_names, "occluders", occluders, "distractors", distractors)

#     reject = []
#     for name in m_names:
#         if (name in occluders) or (name in distractors):
#             reject.append(True)
#         else:
#             reject.append(False)

#     #     object_ids = np.array(f['static']['object_ids'])

#     object_ids = np.array(f["static"]["object_ids"])

#     object_id_to_ind = {}
#     for ct, obj_id in enumerate(object_ids):
#         object_id_to_ind[obj_id] = ct

#     # o_id = np.arange(len(object_ids))

#     id_exchange = np.where(object_ids == np.array(f["static"]["target_id"]))[0].item()
#     special_ones = ["Contain", "Link", "Support"]
#     # if scenario in special_ones:
#     #     o_id[id_exchange] = 1
#     #     o_id[1] = id_exchange

#     # print("object_ids", object_ids)
#     # print("reject", reject)
#     # print("np.array(f['static']['model_names'])", np.array(f['static']['model_names']))

#     colors = [[0, 1, 0], [1, 0, 0]]

#     colors += list(getDistinctColors(len(object_ids)))

#     agent = np.array(f["static"]["target_id"]).item()
#     patient = np.array(f["static"]["zone_id"]).item()

#     colors[object_id_to_ind[patient]] = [0, 1, 0]
#     colors[object_id_to_ind[agent]] = [1, 0, 0]

#     # NOTE THE FIRST OBJECT IS ALWAYS THE PATIENT
#     frame_no = 0

#     # %matplotlib notebook
#     # %matplotlib notebook

#     # %matplotlib inline

#     all_imgs = []

#     all_encodings = []

#     # all_edges_ = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
#     #
#     # all_faces_ = [(0, 1, 2, 3), (4, 5, 6, 7), (2, 3, 6, 7), (1, 0, 4, 5), (1, 2, 6, 5), \
#     #              (0, 4, 7, 3)]

#     pts = [
#         (-1, -1, -1),
#         (-1, -1, 1),
#         (-1, 1, 1),
#         (-1, 1, -1),
#         (1, -1, -1),
#         (1, -1, 1),
#         (1, 1, 1),
#         (1, 1, -1),
#     ]

#     mesh_seq = []

#     for frame_no in indices:  # range(0, len(f['frames']), subsample):
#         frame = str(frame_no).zfill(4)

#         all_obj_pts = []

#         all_meshes = []

#         for ct, obj_id in enumerate(object_ids):
#             #             obj_id = obj_id - 1

#             # TODO: optimize this.
#             if not reject[ct]:
#                 # if scenario in special_ones:
#                 #     ob = obj_id
#                 # else:
#                 ob = ct  # object_id_to_ind[obj_id]

#                 # print(object_ids, obj_id)

#                 vertices_orig, vertices_orig_unscaled, faces_orig = get_vertices_scaled(
#                     f, obj_id, ob
#                 )

#                 # print()

#                 size = np.max(vertices_orig, 0) - np.min(vertices_orig, 0)

#                 size /= 2

#                 size = np.clip(size, -2, 2)

#                 rotations = np.array(
#                     f["frames"][frame]["objects"]["rotations" + prefix][ob]
#                 )
#                 trans = np.array(
#                     f["frames"][frame]["objects"]["positions" + prefix][ob]
#                 )

#                 all_pts, all_edges, all_faces = get_full_bbox(vertices_orig)

#                 # print("all_pts", all_pts.mean(0))

#                 frame_pts = get_transformed_pts(all_pts, rotations, trans)

#                 vertices_orig_transformed = get_transformed_pts(
#                     vertices_orig, rotations, trans
#                 )

#                 mesh = trimesh.Trimesh(
#                     vertices=vertices_orig_transformed, faces=faces_orig
#                 )

#                 all_meshes.append(mesh)

#                 vertices_orig = vertices_orig - np.mean(all_pts, 0, keepdims=True)

#                 vertices_orig /= size[None, :]

#                 centroid = np.mean(frame_pts, 0)

#                 euler_angles = get_rotation(frame_pts)

#                 if len(vertices_orig_unscaled) > 512:
#                     idxx = np.random.randint(0, vertices_orig_unscaled.shape[0], 512)
#                     vertices_orig_unscaled = vertices_orig_unscaled[idxx]
#                     vertices_orig = vertices_orig[idxx]
#                     vertices_orig_transformed = vertices_orig_transformed[idxx]
#                 elif len(vertices_orig_unscaled) < 512:
#                     arr = np.zeros([512 - vertices_orig_unscaled.shape[0], 3])
#                     vertices_orig_unscaled = np.concatenate(
#                         [vertices_orig_unscaled, arr]
#                     )
#                     vertices_orig = np.concatenate([vertices_orig, arr])
#                     vertices_orig_transformed = np.concatenate(
#                         [vertices_orig_transformed, arr]
#                     )

#                 # print("vertices_orig_unscaled", vertices_orig_unscaled.shape)

#                 vertices_orig_unscaled = vertices_orig_unscaled.reshape(-1)
#                 vertices_orig = vertices_orig.reshape(-1)
#                 vertices_orig_transformed = vertices_orig_transformed.reshape(-1)

#                 # print("vertices_orig_transformed", vertices_orig_transformed.shape)

#                 mass_ = [mass[ob]]

#                 encoding = [centroid, euler_angles, size]

#                 if save_mass:
#                     encoding += [mass_]

#                 if save_mesh:
#                     encoding += [vertices_orig]

#                 all_obj_pts.append(np.array(np.concatenate(encoding)))

#                 # all_pts_ = decode_numpy(centroid, euler_angles, size, pts)

#         #             break

#         #         break
#         all_obj_pts = np.stack(all_obj_pts, 0)

#         # print("success")

#         mesh_seq.append(all_meshes)

#         all_encodings.append(all_obj_pts)

#     all_encodings = np.stack(all_encodings, 0)

#     # print("all_encodings", all_encodings.shape)

#     if return_mesh:
#         return all_encodings, mesh_seq

#     return all_encodings


# def get_n_objs_from_encoding(encoding):
#     n_objs = []
#     for enc in encoding:
#         n_objs.append(enc.shape[1])

#     n_objs = torch.tensor(n_objs).to(encoding[0].device)

#     return n_objs


# def unpack_fullnobj_tensor(tensor, n_objs):
#     total_objs = 0
#     output = []
#     for x in n_objs:
#         output.append(tensor[:, total_objs : total_objs + x, :])
#         total_objs += x
#     return output


# FLEX_MASSES = {
#     "triangular_prism": 0.46650617387911586,
#     "torus": 0.13529902128268745,
#     "sphere": 0.5058756912820712,
#     "pyramid": 0.3749996282020332,
#     "platonic": 0.3170116821661435,
#     "pipe": 0.334848007433621,
#     "pentagon": 0.5944103105417653,
#     "octahedron": 0.4999997582702834,
#     "dumbbell": 0.8701171939416522,
#     "cylinder": 0.7653673769319456,
#     "cube": 1.0,
#     "cone": 0.2582547675685456,
#     "bowl": 0.0550834555398029,
# }

# dynamic_friction = 0.25
# static_friction = 0.4
# bounciness = 0.4
# density = 5

# TRIMESH_MESHES = {}

# for key in FLEX_MASSES.keys():
#     TRIMESH_MESHES[key] = None

# TRIMESH_MESHES[key] = trimesh.load(os.path.join(mesh_folder, key + ".obj"))
# mesh = TRIMESH_MESHES[record.name]
# vertices = np.copy(mesh.vertices)
# faces = mesh.faces

# vertices[:, 0] *= scale_factor["x"]
# vertices[:, 1] *= scale_factor["y"]
# vertices[:, 2] *= scale_factor["z"]
# scaled_mesh_volume = trimesh.Trimesh(vertices, faces).volume
# mass = scaled_mesh_volume * density

# commands.extend(
#     [
#         {"$type": "set_mass", "mass": mass, "id": object_id},
#         {
#             "$type": "set_physic_material",
#             "dynamic_friction": dynamic_friction,
#             "static_friction": static_friction,
#             "bounciness": bounciness,
#             "id": object_id,
#         },
#     ]
# )
