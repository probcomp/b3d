import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import rerun as rr
from b3d.chisight.sparse.gps_utils import cov_from_dq_composition
from b3d.io import MeshData
from b3d.pose import Pose
from b3d.utils import keysplit
from jax.scipy.spatial.transform import Rotation as Rot
from sklearn.utils import Bunch


# **************************
#   Mesh Subsampling
#   Methods
# **************************
def area_of_triangle(vs):
    """
    Computes the area of a triangle spanned by 3 stacked
    vectors (i.e. the rows of the array contain the vertices).
    """
    a = vs[1] - vs[0]
    b = vs[2] - vs[0]
    na = jnp.linalg.norm(a)
    nb = jnp.linalg.norm(b)
    alpha = jnp.arccos(jnp.dot(a / na, b / nb))
    return 1 / 2 * jnp.sin(alpha) * na * nb


def uniform_on_faces(key, fs, probs, num):
    """
    Args:
        key: Random Key
        fs: Array of triangles (faces) of shape (M,3)
        probs: Probabilities over faces
        num: Number of samples

    Returns:
        Pair of face indices and barrycentric coordinates on each face.
    """
    ii = jax.random.choice(key, jnp.arange(fs.shape[0]), p=probs, shape=(num,))
    ps = jax.random.dirichlet(key, jnp.ones(3), shape=(num,))
    return ii, ps


def put_on_mesh(i, p, t, fs, vs):
    return jnp.sum(p[:, :, None] * vs[t, fs[i]], axis=1)


def subsample_and_track(key, N, vs, fs, t=0):
    """
    Args:
        key: Random Key
        N: Number of subsamples
        vs: Array of time dependent vertices of shape (T,K,3)
        fs: Array of triangles (faces) of shape (M,3)
    """
    areas = jax.vmap(area_of_triangle)(vs[t, fs])
    triangle_probs = areas / areas.sum()
    ii, pp = uniform_on_faces(key, fs, triangle_probs, N)
    xs = jax.vmap(lambda t: put_on_mesh(ii, pp, t=t, fs=fs, vs=vs))(
        jnp.arange(vs.shape[0])
    )
    return xs


def embedding_matrix_from_dq_composition(diags, quat):
    U = Rot.from_quat(quat).as_matrix()
    D = U @ jnp.diag(jnp.sqrt(diags)) @ U.T
    return U @ D @ U.T


# **************************
#   Mesh Visuals
# **************************
def combine_meshes(ms):
    vs = []
    fs = []
    off = 0
    for v, f in ms:
        fs.append(f + off)
        vs.append(v)
        off += len(v)

    return np.concatenate(vs), np.concatenate(fs)


def create_sphere_mesh(m, r=1.0):
    """
    Generates a 3D mesh for a unit sphere centered at the origin.
    Returns vertices, faces, and vertex normals.
    """

    def _angles_to_xyz(angles):
        theta, phi = angles
        return np.array(
            [np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)]
        )

    vertices = []
    faces = []
    normals = []

    n = m + m - 1
    angles = np.stack(
        np.meshgrid(
            np.linspace(0, 2 * np.pi, n)[:-1],
            np.linspace(-np.pi / 2, np.pi / 2, m),
            indexing="xy",
        ),
        axis=-1,
    )
    vertices = r * np.apply_along_axis(_angles_to_xyz, 2, angles).reshape(-1, 3)
    normals = vertices

    shift = np.arange(n - 1) + 1
    shift[n - 2] = 0

    face_row = np.concatenate(
        [
            np.stack(
                [np.arange(n - 1), shift, shift + n - 1],
                axis=-1,
            ),
            np.stack(
                [np.arange(n - 1), shift + n - 1, np.arange(n - 1) + n - 1],
                axis=-1,
            ),
        ]
    )
    faces = np.concatenate([face_row + i * (n - 1) for i in range(m - 1)])

    return vertices, faces, normals


def sphere_mesh(x=jnp.zeros(3), r=1.0, c=None, segs=5):
    vertices, faces, _ = create_sphere_mesh(segs, r=r)

    if c is not None:
        n = len(vertices)
        cs = np.tile(c, (n, 1))
    else:
        cs = None

    return Bunch(
        vertex_positions=vertices + x[None],
        indices=faces,
        vertex_normals=vertices,
        vertex_colors=cs,
    )


def gps_mesh_with_A(xs, As, r=1.0, cs=None, segs=10):
    xs = np.array(xs)
    n = xs.shape[0]
    v0, f0, _ = create_sphere_mesh(segs, r=r)
    nv = v0.shape[0]

    vs = np.stack([v0 @ A.T for A in As], axis=0)
    fs = np.tile(f0, (n, 1, 1))
    ns = vs.copy()
    vs += xs[:, None, :]
    fs += (np.arange(n) * nv)[:, None, None]

    if cs is not None:
        cs = cs[:, None, :] * jnp.ones((n, nv, 1))
        cs = cs.reshape(-1, 3)

    return Bunch(
        vertex_positions=vs.reshape(-1, 3),
        indices=fs.reshape(-1, 3),
        vertex_normals=ns.reshape(-1, 3),
        vertex_colors=cs,
    )


def gps_mesh(xs, r=1.0, cs=None, segs=10):
    xs = np.array(xs)
    n = xs.shape[0]
    v0, f0, _ = create_sphere_mesh(segs, r=r)
    nv = v0.shape[0]

    vs = np.tile(v0, (n, 1, 1))
    fs = np.tile(f0, (n, 1, 1))
    ns = vs.copy()
    vs += xs[:, None, :]
    fs += (np.arange(n) * nv)[:, None, None]

    if cs is not None:
        cs = cs[:, None, :] * jnp.ones((n, nv, 1))
        cs = cs.reshape(-1, 3)

    return Bunch(
        vertex_positions=vs.reshape(-1, 3),
        indices=fs.reshape(-1, 3),
        vertex_normals=ns.reshape(-1, 3),
        vertex_colors=cs,
    )


# **************************
#   Load and Subsample
#   Mesh Data
# **************************10
DEFAULT_PATH = "../dcolmap/assets/shared_data_bucket/input_data/unity/deformablemesh"
# DEFAULT_PATH = Path("./")
path = Path(
    input(f"Type Data directory \n(`{DEFAULT_PATH}` Default): ").strip() or DEFAULT_PATH
)
files = os.listdir(path)
print("Listing files from directory...")
for i, f in enumerate(files):
    print(f"{i}:", f)

i = int(
    input("Type the index of the file you want to load (0 Default): ").strip() or "0"
)
fname = path / files[i]
data = MeshData.load(fname)
_vs = data.vertices_positions
_fs = data.triangles
_T = _vs.shape[0]
_N = _vs.shape[1]

print(
    f"""
Loaded mesh data from
    {fname}

Num timesteps: {_T}
Num particles: {_N}
_vs.shape: {_vs.shape}
_fs.shape: {_fs.shape}
"""
)

# Create subsamples from the mesh
# and track over time
N = int(input("Number of Subsamples \n(500 default): ").strip() or "500")
T = int(input(f"Number of Timesteps \n({_T} default): ").strip() or f"{_T}")
N = 500
key = jax.random.PRNGKey(0)
key = keysplit(key)
xs = subsample_and_track(key, N, _vs[:T], _fs)

print(
    f"""
Subsampled mesh data:
    xs.shape: {xs.shape}
    scale = {(xs[0].max(0) - xs[0].min(0)).max()}
"""
)
# scale = xs.var(1).max()
scale = 0.25 * (xs[0].max(0) - xs[0].min(0)).max()
xs = xs - xs[0].mean(0)[None, None]
xs = xs / scale


# **************************
#   Data fitting and
#   Optimization Loop
# **************************
logsumexp = jax.scipy.special.logsumexp


def loss(xs, relative_poses, diagonal_covariances, cluster_poses):
    """
    Args:
        xs: 3D-Keypoint positions over time; Array of shape (T, N, 3)
        relative_poses: Particle pose in cluster coordinates; Pose Array of shape (N, )
        diagonal_covariances: Diagonal covariances of the particles; Array of shape (N, 3)
        cluster_poses: Cluster pose over time of a single cluster; Pose Array of shape (T, )
    """
    particle_pose_tracks = cluster_poses[:, None] @ relative_poses

    mus = particle_pose_tracks.pos
    covs = jax.vmap(jax.vmap(cov_from_dq_composition), (None, 0))(
        diagonal_covariances, particle_pose_tracks.quat
    )

    logp = jax.scipy.stats.multivariate_normal.logpdf(xs, mus, covs).sum()

    return -logp


# Parameter dict with with initial values
# that will be optimized
cluster_poses = Pose.from_pos(xs.mean(1))
absolute_poses = Pose.from_pos(xs.mean(0))
relative_poses = cluster_poses[0].inv() @ absolute_poses
diagonal_covariances = 0.1 * jnp.ones((N, 3))

params = dict(
    relative_poses=relative_poses,
    diagonal_covariances=diagonal_covariances,
    cluster_poses=cluster_poses,
)


# Setting up the optimization
# NOTE: xs are baked-in here.
def loss_func(params):
    return loss(xs, **params)


grad_loss = jax.value_and_grad(loss_func)
optimizer = optax.adam(1e-4)
opt_state = optimizer.init(params)


@jax.jit
def step(carry, _):
    params, opt_state = carry
    ell, grads = grad_loss(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    params["diagonal_covariances"] = jnp.clip(
        params["diagonal_covariances"], 1e-3, jnp.inf
    )
    return ((params, opt_state), ell)


# **************************
#   Run the Loop
# **************************
print("Fitting data...")
num_runs = 20
num_runs = int(
    input(f"Number of training iteration loops \n({num_runs} default): ").strip()
    or f"{num_runs}"
)
for i in range(num_runs):
    (params, opt_state), losses = jax.lax.scan(
        step, (params, opt_state), xs=None, length=500
    )
    print(f"Iteration {i+1}/{num_runs}, Average Loss: {losses.mean():,.5}")
print("...done!")

# **************************
#   Visualize the result
# **************************
print("""
    Creating rerun visualization
""")
PORT = 8812
rr.init("Deformable")
rr.connect(addr=f"127.0.0.1:{PORT}")

ps = params["relative_poses"]
qs = params["cluster_poses"]
diags = params["diagonal_covariances"]
ps_world = qs[:, None] @ ps


rrid = "Demo"
# for t in np.arange(xs.shape[0], step=5):
for t in np.arange(5, step=1):
    rr.set_time_sequence("frame_idx", t)

    # Offset to center the visualization
    off = qs[t].pos[None]

    # Embedding matrices that describe
    # the covariance ellipsoids
    embedding_matrices = jax.vmap(embedding_matrix_from_dq_composition)(
        diags, ps_world[t].quat
    )

    rr.log(
        f"{rrid}/particle_position",
        rr.Mesh3D(
            **gps_mesh(ps_world[t].pos - off, r=0.05),
            mesh_material=rr.components.Material(albedo_factor=[200, 200, 200]),
        ),
    )
    rr.log(
        f"{rrid}/particle_covariance",
        rr.Mesh3D(
            **gps_mesh_with_A(ps_world[t].pos - off, embedding_matrices, r=1.0),
            mesh_material=rr.components.Material(albedo_factor=[200, 200, 200]),
        ),
    )
    rr.log(
        f"{rrid}/data",
        rr.Mesh3D(
            **gps_mesh(xs[t] - off, r=0.05),
            mesh_material=rr.components.Material(albedo_factor=[255, 0, 0]),
        ),
    )
