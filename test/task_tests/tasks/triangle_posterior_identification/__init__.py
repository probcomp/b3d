from test.task_tests.common import Task
import jax.numpy as jnp
import b3d
import rerun as rr

class TrianglePosteriorIdentificationTask(Task):
    """
    """
    def __init__(self,
                 scene_background, foreground_triangle, triangle_path, camera_path,
                 renderer=None
                ):
        """
        scene_background: (vertices, faces, face_colors) for background
        foreground_triangle: dict with keys "color" (rgb array) and "vertices" (3x3 array)
        triangle_path: 3D path of the triangle. Batched Pose object.
        camera_path: 3D path of the camera. Batched Pose object.
        """
        self.scene_background = scene_background
        self.foreground_triangle = foreground_triangle
        self.triangle_path = triangle_path
        self.camera_path = camera_path

        if renderer is None:
            renderer = TrianglePosteriorIdentificationTask.get_default_renderer()
        self.renderer = renderer

        self._rr_initialized = False

    def get_test_pair(self):
        video = self.get_video()
        tri_color = self.foreground_triangle["color"]
        tri_shape = self._get_triangle_angles()
        initial_2d_point = self._get_projection_of_triangle_center()
        task_input = (video, (tri_color, tri_shape, initial_2d_point))
        other_input_to_scorer = None # no loading needed -- can just get the info off `self`

        return task_input, other_input_to_scorer

    def score(self, task_input, baseline, solution):
        """
        The `solution` should be a data structure containing different
        algorithms for approximating the posterior.
        It can contain the following keys:
        - "laplace"
        - "grid"
        - "mala"
        - "multi_initialized_mala"
        """
        return {
            "laplace": self.score_laplace_approximation(task_input, baseline, solution["laplace"]),
            "grid": self.score_grid_approximation(task_input, baseline, solution["grid"]),
            "mala": self.score_mala(task_input, baseline, solution["mala"]),
            "multi_initialized_mala": self.score_multi_initialized_mala(task_input, baseline, solution["multi_initialized_mala"])
        }

    def assert_passing(self, tester, metrics):
        pass

    ### Sub-tests ###
    def score_laplace_approximation(self, task_input, baseline, solution):
        """
        solution: (mean, covariance) tuple for a laplace approximation to the
        distribution over the triangle's 3D pose as a 7D [*pos, *xyzw] vector.
        """
        pass

    def score_grid_approximation(self, task_input, baseline, solution):
        pass

    def score_mala(self, task_input, baseline, solution):
        pass

    def score_multi_initialized_mala(self, task_input, baseline, solution):
        pass

    ### Helpers ###
    def n_frames(self):
        return self.triangle_path.shape[0]
    
    def get_camera_frame_scene(self, frame_idx):
        # background
        b_vertices, b_faces, b_triangle_colors = self.scene_background
        # foreground
        f_vertices = self.foreground_triangle["vertices"]
        f_colors = jnp.array([self.foreground_triangle["color"]])
        # to camera frame
        camera_pose = self.camera_path[frame_idx]
        b_vertices = camera_pose.inv().apply(b_vertices)
        f_vertices = camera_pose.inv().apply(f_vertices)
        # unified vertices/faces/colors
        vertices = jnp.concatenate([b_vertices, f_vertices], axis=0)
        faces = jnp.concatenate([b_faces,
                    jnp.arange(3).reshape((1, 3)) + b_vertices.shape[0]], axis=0)
        triangle_colors = jnp.concatenate([b_triangle_colors, f_colors], axis=0)
        return (vertices, faces, triangle_colors)

    def get_image_at_frame(self, frame_idx):
        scene = self.get_camera_frame_scene(frame_idx)
        vertices, faces, vertex_colors = b3d.triangle_color_mesh_to_vertex_color_mesh(*scene)
        image, _ = self.renderer.render_attribute(
            b3d.Pose.identity()[None, ...], vertices, faces,
            jnp.array([[0, len(faces)]]), vertex_colors
        )
        return image
    
    def get_video(self):
        return jnp.stack([self.get_image_at_frame(i) for i in range(self.n_frames())])

    def visualize_scene(self):
        assert self.n_frames() == 1, "Current visualizer only configured for single-frame scenes"
        self.maybe_initialize_rerun()

        v, f, vc = b3d.triangle_color_mesh_to_vertex_color_mesh(*self.scene_background)
        rr.log("/3D/scene/background", rr.Mesh3D(
            vertex_positions=v, indices=f, vertex_colors=vc
        ))
        rr.log("/3D/scene/foreground", rr.Mesh3D(
            vertex_positions=self.foreground_triangle["vertices"],
            indices=jnp.arange(3).reshape((1, 3)),
            vertex_colors=jnp.tile(self.foreground_triangle["color"], (3, 1))
        ))

        r = self.renderer
        rr.log("/3D/scene/camera", rr.Pinhole(
            focal_length=r.fx, width=r.width, height=r.height),
            timeless=True
        )
        rr.log("/3D/scene/camera", rr.Transform3D(
            translation=self.camera_path[0].position,
            rotation=rr.Quaternion(xyzw=self.camera_path[0].xyzw)
        ))
        rr.log("/3D/scene/camera", rr.Image(self.get_image_at_frame(0)), timeless=True)


    # TODO: is there a way to get the task config `name` in here?
    def maybe_initialize_rerun(self):
        if not self._rr_initialized:
            rr.init("triangle_posterior_identification")
            rr.connect("127.0.0.1:8812")
            self._rr_initialized = True

    @staticmethod
    def get_monochrome_room_background(rgb):
        vertices = jnp.array([
            [0, 0, 0],
            [0, 0, 18],
            [2, 4, 0],
            [2, 4, 18],
            [9, 4, 0],
            [9, 4, 18],
            [11, 0, 0],
            [11, 0, 18]
        ], dtype=float)
        faces = jnp.array([
            [0, 1, 3],
            [0, 3, 2],
            [2, 3, 5],
            [2, 5, 4],
            [4, 5, 7],
            [4, 7, 6]
        ])
        face_colors = jnp.tile(rgb, (6, 1))
        return (vertices, faces, face_colors)
    
    @staticmethod
    def get_default_renderer():
        image_width = 120; image_height = 100; fx = 50.0; fy = 50.0
        cx = 50.0; cy = 50.0; near = 0.001; far = 16.0
        return b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)
