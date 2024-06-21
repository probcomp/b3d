import b3d.chisight.dense.differentiable_renderer as differentiable_renderer
from ..common.task import Task
import jax
import jax.numpy as jnp
import b3d
import rerun as rr
import matplotlib.pyplot as plt

class TrianglePosteriorGridApproximationTask(Task):
    """
    This task presents an image of a triangle, and asks for the posterior over the pose of the underlying
    triangle in the scene.
    It asks for an approximation to the posterior over the depth of the triangle, given the image.
    It queries this by asking for the approximate probability that the triangle's first vertex
    is at depth in range [a, b], for a set of depth ranges forming a partition of an interval
    of depth space.
    """

    ### Constructor ###

    def __init__(self,
                 scene_background, foreground_triangle, triangle_path, camera_path,
                 renderer=None,
                 taskname=""
                ):
        """
        scene_background: (vertices, faces, face_colors) for background
            vertices: (N, 3) array of vertex positions
            faces: (M, 3) array of vertex indices
            face_colors: (M, 3) array of rgb colors
        foreground_triangle: dict with keys "color" (rgb array) and "vertices" (3x3 array)
        triangle_path: 3D path of the triangle. Batched Pose object.
        camera_path: 3D path of the camera. Batched Pose object.
        """
        self.scene_background = scene_background
        self.foreground_triangle = foreground_triangle
        self.triangle_path = triangle_path
        self.camera_path = camera_path

        if renderer is None:
            renderer = TrianglePosteriorGridApproximationTask.get_default_renderer()
        self.renderer = renderer

        self.taskname = taskname
        self._rr_initialized = False

    ### Task interface ###

    def get_task_specification(self):
        video = self.get_video()
        
        # Remove ground truth depth and size information from triangle,
        # by making the first vertex [0, 0, 0], and making
        # the edge v0->v1 have length 1
        triangle = self.foreground_triangle["vertices"]
        triangle = triangle - triangle[0]
        triangle = triangle / jnp.linalg.norm(triangle[1])

        partition, background_z = self._get_grid_partition()

        task_spec = {
            "video": video,
            "triangle": {
                "vertices": triangle,
                "color": self.foreground_triangle["color"]
            },
            "background_mesh": self.scene_background,
            "renderer": self.renderer,
            "camera_path": self.camera_path,
            "depth_partition": partition,
            "cheating_info": {
                "triangle_vertices": self.foreground_triangle["vertices"]
            }
        }

        return task_spec

    def score(self, solution, **kwargs):
        if self.n_frames() == 1:
            return self.score_oneframe_grid_approximation(solution)
        else:
            return {}

    def assert_passing(self, metrics,
        mass_outside_region_threshold = 0.001,
        divergence_from_uniform_threshold = 0.2
    ):
        assert metrics["mass_assigned_outside_expected_region"] <= mass_outside_region_threshold
        assert metrics["divergence_from_uniform_in_expected_region"] <= divergence_from_uniform_threshold

    def visualize_task(self):
        self.visualize_scene()

    def visualize_solution(self, solution, metrics):
        partition, _ = self._get_grid_partition()
        self.visualize_grid_approximation(jnp.exp(metrics["log_posterior_approximation"]), show_expected_region=False)
        self.visualize_grid_approximation(jnp.exp(metrics["expected_log_posterior"]), name="expected_grid_posterior")
        self.visualize_log_grid_approximation_2D(partition, metrics["log_posterior_approximation"], name="log_grid_posterior_approximation")
        self.visualize_log_grid_approximation_2D(partition, metrics["expected_log_posterior"], name="expected_log_grid_posterior")
    
    ### Scoring implementation ###

    def score_oneframe_grid_approximation(self, solution):
        partition, background_z = self._get_grid_partition()
        log_posterior_approximation = solution
        posterior_approximation = jnp.exp(log_posterior_approximation)

        # STEP 3: compute the metrics
        max_idx_before_region = jnp.argmax(0. < partition)
        min_idx_after_region = jnp.argmax(background_z < partition)
        mass_assigned_outside_expected_region = \
            jnp.sum(posterior_approximation[:max_idx_before_region]) + \
            jnp.sum(posterior_approximation[min_idx_after_region:])
        posterior_in_expected_region = posterior_approximation[max_idx_before_region:min_idx_after_region]
        posterior_in_expected_region = posterior_in_expected_region / jnp.sum(posterior_in_expected_region)
        uniform_in_expected_region = jnp.ones_like(posterior_in_expected_region) / len(posterior_in_expected_region)
        divergence_from_uniform = _kl(
            uniform_in_expected_region,
            posterior_in_expected_region
        )
        expected_posterior = jnp.zeros_like(partition[:-1])
        expected_posterior = expected_posterior.at[max_idx_before_region:min_idx_after_region].set(uniform_in_expected_region)
        expected_log_posterior = jnp.log(expected_posterior)
        
        return {
            "mass_assigned_outside_expected_region": mass_assigned_outside_expected_region,
            "divergence_from_uniform_in_expected_region": divergence_from_uniform,
            "log_posterior_approximation": log_posterior_approximation,
            "expected_log_posterior": expected_log_posterior
        }
        
    def _get_grid_partition(self):
        X_WC = self.camera_path[0]
        triangle_C = X_WC.inv().apply(self.foreground_triangle["vertices"])
        background_vertices_C = X_WC.inv().apply(self.scene_background[0])
        background_faces = self.scene_background[1]
        def get_int_z(face_idx):
            int_pt = differentiable_renderer.project_ray_to_plane(jnp.zeros(3), triangle_C[0], background_vertices_C[background_faces[face_idx]])
            # print(int_pt.shape)
            # print(background_vertices_C[background_faces[face_idx]].shape)
            is_in_triangle, _ = differentiable_renderer.pt_is_in_plane(background_vertices_C[background_faces[face_idx]], int_pt)
            # set z to infinity if the intersection point is not in the triangle
            return jnp.where(
                is_in_triangle,
                int_pt[2],
                jnp.inf
            )
        int_pts_C = jax.vmap(get_int_z)(jnp.arange(background_faces.shape[0]))
        background_z = jnp.min(int_pts_C)
        jax.experimental.checkify.check(not jnp.isinf(background_z), "No intersection found")

        partition = jnp.linspace(- 1.0, background_z + 1.0, 100)
        return partition, background_z

    ### Helpers ###

    @classmethod
    def default_scene_using_colors(cls, background_color, triangle_color):
        return cls(**cls.generate_unichromatic_background_single_frame_taskargs(
            background_color, triangle_color
        ))

    @classmethod
    def generate_unichromatic_background_single_frame_taskargs(
        cls, background_color, triangle_color, taskname=None
    ):
        return {
                "scene_background": cls.get_monochrome_room_background(
                    background_color
                ),
                "foreground_triangle": {
                    "color": triangle_color,
                    "vertices": jnp.array([
                        [5, 3, 7], [6, 1.5, 7], [6, 1.5, 8]
                    ], dtype=float)
                },
                "triangle_path": b3d.Pose.identity()[None, ...],
                "camera_path": b3d.pose.camera_from_position_and_target(
                    position=jnp.array([5., -0.5, 6.]),
                    target=jnp.array([5, 3, 7.])
                )[None, ...],
                "taskname": taskname
            }

    def n_frames(self):
        return self.triangle_path.shape[0]
    
    def visualize_log_grid_approximation_2D(self, partition, log_posterior_approximation, name="log_grid_posterior_approximation"):
        plt.plot(partition[:-1], log_posterior_approximation, label=name)
        plt.title("Log grid posterior approximation")
        plt.xlabel("depth")
        plt.ylabel("log P(depth | data)")


    def visualize_grid_approximation(self, posterior_approximation, name="grid_posterior_approximation", show_expected_region=True):
        self.maybe_initialize_rerun()
        partition, background_z = self._get_grid_partition()
        pts_C = partition[:, None] * jnp.array([[0., 0., 1.]])
        X_WC = self.camera_path[0]
        pts_W = X_WC.apply(pts_C)

        def p_to_color(p):
            c = jnp.sqrt(p * 10)
            c = jnp.clip(c, 0., 1.)
            return c

        rr.log(
            f"/3D/{name}/camera_z_partition/partition",
            rr.LineStrips3D(
                [jnp.array([pts_W[i], pts_W[i+1]]) for i in range(len(pts_W) - 1)],
                colors=[p_to_color(a) for a in jnp.clip(posterior_approximation, 0., 1.)]
            )
        )

        if show_expected_region:
            max_idx_before_region = jnp.argmax(0. < partition)
            min_idx_after_region = jnp.argmax(background_z < partition)
            rr.log(
                f"/3D/{name}/camera_z_partition/possible_region",
                rr.LineStrips3D(
                    [[pts_W[max_idx_before_region + 1], pts_W[min_idx_after_region - 1]]],
                    colors=[[0., 1., 1.]]
                )
            )

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
        vertices, faces, vertex_colors = b3d.utils.triangle_color_mesh_to_vertex_color_mesh(*scene)
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

        v, f, vc = b3d.utils.triangle_color_mesh_to_vertex_color_mesh(*self.scene_background)
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
            rr.init(f"triangle_posterior_identification--{self.taskname}")
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

def _kl(p, q):
    return jnp.sum(p * jnp.log(p / q))