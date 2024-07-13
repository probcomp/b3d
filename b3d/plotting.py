import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn.utils import Bunch
from PIL import Image
import io
import jax
import jax.numpy as jnp
from IPython.display import Image as IPImage, display


# **************************
#   GIF and other Animations
# **************************

def fig_to_image(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    return img


def save_as_gif(fname, images, fps=10, loop=0):
    """Save a list of images as a gif"""
    if isinstance(images[0], np.ndarray) or isinstance(images[0], jnp.ndarray):
        images = [Image.fromarray(im) for im in images]

    if not isinstance(images[0], Image.Image):
        raise Exception("images need to be `(j)numpy.ndarray` or `PIL.Image.Image`")

    images[0].save(
        fname,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=1000.0 / fps,
        loop=loop,
    )

    return fname


def display_gif(fname):
    return display(IPImage(data=open(fname, "rb").read(), format="png"))


def save_and_display_gif(fname, images, fps=10, loop=0):
    return display_gif(save_as_gif(fname, images, fps=fps, loop=loop))

# **************************
#   Matplotlib
# **************************
from matplotlib.collections import LineCollection


def adjust_angle(hd):
    """Adjusts angle to lie in the interval [-pi,pi)."""
    return (hd + jnp.pi)%(2*jnp.pi) - jnp.pi

def line_collection(a, b, c=None, linewidth=1, **kwargs):
    lines = np.column_stack((a, b)).reshape(-1, 2, 2)
    lc = LineCollection(lines, colors=c, linewidths=linewidth, **kwargs)
    return lc

def plot_segs(segs, c="k", linewidth=1, ax=None,  **kwargs):
    if ax is None: ax = plt.gca()
    n = 10
    segs = segs.reshape(-1,2,2)
    a = segs[:,0]
    b = segs[:,1]
    lc = line_collection(a, b, linewidth=linewidth, **kwargs)
    lc.set_colors(c)
    ax.add_collection(lc)

# **************************
#   Mesh Helper
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


def create_sphere_mesh(mesh_resolution=10, r=1.0):
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

    m = mesh_resolution
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


def sphere_mesh(x=jnp.zeros(3), r=1.0, c=None, mesh_resolution=10):
    vertices, faces, _ = create_sphere_mesh(mesh_resolution, r=r)

    if c is not None:
        n = len(vertices)
        cs = np.tile(c, (n, 1))
    else:
        cs = None

    return Bunch(
        vertex_positions=vertices + x[None],
        triangle_indices=faces,
        vertex_normals=vertices,
        vertex_colors=cs,
    )


from b3d.chisight.gps_utils import ellipsoid_embedding, cov_from_dq_composition


def gaussian_mesh_from_xs_and_covs(xs, covs, r=1.0, cs=None, mesh_resolution=10):
    n = xs.shape[0]

    embs = jax.vmap(ellipsoid_embedding)(covs)

    v0, f0, _ = create_sphere_mesh(mesh_resolution, r=r)
    nv = v0.shape[0]
    nf = f0.shape[0]

    vs = np.stack([v0 @ A.T for A in embs], axis=0)
    fs = np.tile(f0, (n, 1, 1))
    ns = vs.copy()
    vs += xs[:, None, :]
    fs += (np.arange(n) * nv)[:, None, None]

    if cs is not None:
        cs = cs[:, None, :] * jnp.ones((n, nv, 1))
        cs = cs.reshape(-1, 3)

    return Bunch(
        vertex_positions=vs.reshape(-1, 3),
        triangle_indices=fs.reshape(-1, 3),
        vertex_normals=ns.reshape(-1, 3),
        vertex_colors=cs,
    )


def gaussian_mesh_from_ps_and_diags(ps, diags, r=1.0, cs=None, mesh_resolution=10):
    covs = jax.vmap(cov_from_dq_composition)(diags, ps.quat)
    xs = ps.pos
    return gaussian_mesh_from_xs_and_covs(
        xs, covs, r=r, cs=cs, mesh_resolution=mesh_resolution)



def gps_mesh(xs, r=1.0, cs=None, mesh_resolution=10):
    xs = np.array(xs)
    n = xs.shape[0]
    v0, f0, _ = create_sphere_mesh(mesh_resolution, r=r)
    nv = v0.shape[0]
    nf = f0.shape[0]

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
        triangle_indices=fs.reshape(-1, 3),
        vertex_normals=ns.reshape(-1, 3),
        vertex_colors=cs,
    )
