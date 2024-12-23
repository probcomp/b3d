import itertools
from functools import partial
import warnings

import genjax
import jax
import jax.numpy as jnp
from genjax import Pytree
from tensorflow_probability.substrates import jax as tfp

from b3d.pose import (
    logpdf_gaussian_vmf_pose,
    logpdf_uniform_pose,
    logpdf_uniform_scale,
    sample_gaussian_vmf_pose,
    sample_uniform_pose,
    sample_uniform_scale,
)


def separating_axis_test(axis, box1, box2):
    """
    Projects both boxes onto the given axis and checks for overlap.
    """
    min1, max1 = project_box(axis, box1)
    min2, max2 = project_box(axis, box2)

    return jax.lax.cond(jnp.logical_or(max1 < min2, max2 < min1), lambda: False, lambda: True)

    # if max1 < min2 or max2 < min1:
    #     return False
    # return True

def project_box(axis, box):
    """
    Projects a box onto an axis and returns the min and max projection values.
    """
    corners = get_transformed_box_corners(box)
    projections = jnp.array([jnp.dot(corner, axis) for corner in corners])
    return jnp.min(projections), jnp.max(projections)

def get_transformed_box_corners(box):
    """
    Returns the 8 corners of the box based on its dimensions and pose.
    """
    dim, pose = box
    corners = []
    for dx in [-dim[0]/2, dim[0]/2]:
        for dy in [-dim[1]/2, dim[1]/2]:
            for dz in [-dim[2]/2, dim[2]/2]:
                corner = jnp.array([dx, dy, dz, 1])
                transformed_corner = pose @ corner
                corners.append(transformed_corner[:3])
    return corners

def are_bboxes_intersecting(dim1, dim2, pose1, pose2):
    """
    Checks if two oriented bounding boxes (OBBs), which are AABBs with poses, are intersecting using the Separating 
    Axis Theorem (SAT).

    Args:
        dim1 (jnp.ndarray): Bounding box dimensions of first object. Shape (3,)
        dim2 (jnp.ndarray): Bounding box dimensions of second object. Shape (3,)
        pose1 (jnp.ndarray): Pose of first object. Shape (4,4)
        pose2 (jnp.ndarray): Pose of second object. Shape (4,4)
    Output:
        Bool: Returns true if bboxes intersect
    """
    box1 = (dim1, pose1)
    box2 = (dim2, pose2)

    # Axes to test - the face normals of each box
    axes_to_test = []
    for i in range(3):  # Add the face normals of box1
        axes_to_test.append(pose1[:3, i])
    for i in range(3):  # Add the face normals of box2
        axes_to_test.append(pose2[:3, i])

    # Perform SAT on each axis
    count_ = 0
    for axis in axes_to_test:
        count_+= jax.lax.cond(separating_axis_test(axis, box1, box2), lambda:0,lambda:-1)

    return jax.lax.cond(count_ < 0, lambda:False,lambda:True)

are_bboxes_intersecting_jit = jax.jit(are_bboxes_intersecting)
# For one reference pose (object 1) and many possible poses for the second object
are_bboxes_intersecting_many = jax.vmap(are_bboxes_intersecting, in_axes = (None, None, None, 0))
are_bboxes_intersecting_many_jit = jax.jit(are_bboxes_intersecting_many)


############################
#### compute the volume ####
############################
@jax.jit
def ray_intersects_triangle(p0, d, v0, v1, v2):
    epsilon = 1e-6
    e1 = v1 - v0
    e2 = v2 - v0
    h = jnp.cross(d, e2)
    a = jnp.dot(e1, h)
    parallel = jnp.abs(a) < epsilon
    f = 1.0 / a
    s = p0 - v0
    u = f * jnp.dot(s, h)
    valid_u = (u >= 0.0) & (u <= 1.0)
    q = jnp.cross(s, e1)
    v = f * jnp.dot(d, q)
    valid_v = (v >= 0.0) & (u + v <= 1.0)
    t = f * jnp.dot(e2, q)
    valid_t = t > epsilon
    intersects = (~parallel) & valid_u & valid_v & valid_t
    return intersects


@jax.jit
def point_in_mesh(point, vertices, faces):
    ray_direction = jnp.array([1.0, 0.0, 0.0])  # Arbitrary direction
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    intersects = jax.vmap(ray_intersects_triangle, in_axes=(None, None, 0, 0, 0))(
        point, ray_direction, v0, v1, v2
    )
    num_intersections = jnp.sum(intersects)
    return num_intersections % 2 == 1  # Inside if odd number of intersections


def min_max_coord(vertices):
    min_coords = jnp.min(vertices, axis=0)
    max_coords = jnp.max(vertices, axis=0)
    return min_coords, max_coords


@partial(jax.jit, static_argnames=["num_samples"])
def monte_carlo_intersection_volume(
    mesh1_vertices, mesh1_faces, mesh2_vertices, mesh2_faces, num_samples, key
):
    min_coords1, max_coords1 = min_max_coord(mesh1_vertices)
    min_coords2, max_coords2 = min_max_coord(mesh2_vertices)

    min_coords = jnp.maximum(min_coords1, min_coords2)
    max_coords = jnp.minimum(max_coords1, max_coords2)

    overlap = jnp.all(min_coords < max_coords)
    bbox_volume = jnp.prod(max_coords - min_coords)

    def sample_points(key):
        subkey_x, subkey_y, subkey_z = jax.random.split(key, 3)
        x = jax.random.uniform(
            subkey_x, shape=(num_samples,), minval=min_coords[0], maxval=max_coords[0]
        )
        y = jax.random.uniform(
            subkey_y, shape=(num_samples,), minval=min_coords[1], maxval=max_coords[1]
        )
        z = jax.random.uniform(
            subkey_z, shape=(num_samples,), minval=min_coords[2], maxval=max_coords[2]
        )
        points = jnp.stack([x, y, z], axis=1)
        return points

    points = sample_points(key)

    point_in_mesh_vmap = jax.vmap(point_in_mesh, in_axes=(0, None, None))

    in_mesh1 = point_in_mesh_vmap(points, mesh1_vertices, mesh1_faces)
    in_mesh2 = point_in_mesh_vmap(points, mesh2_vertices, mesh2_faces)
    in_both_meshes = in_mesh1 & in_mesh2

    hits = jnp.sum(in_both_meshes)
    intersection_volume = (hits / num_samples) * bbox_volume * overlap
    return intersection_volume


def get_interpenetration(mesh_seq, num_samples):
    interpenetrations = []
    for ct, pair in enumerate(list(itertools.combinations(mesh_seq, 2))):
        m1, m2 = pair
        # Monte Carlo parameters
        key = jax.random.PRNGKey(ct)  # Random seed
        # Compute intersection volume
        intersection_volume = monte_carlo_intersection_volume(
            m1.vertices, m1.faces, m2.vertices, m2.faces, num_samples, key
        )
        interpenetrations.append(intersection_volume)

    # floor interpenetration approximation
    for mesh in mesh_seq:
        bottom = mesh.vertices[:, 1].min()
        interpenetrations.append(jnp.where(bottom < 0, abs(bottom), 0))
        # if bottom < 0:
        #     interpenetrations.append(abs(bottom))
    return jnp.array(interpenetrations).sum()


@jax.jit
def sample_uniform_broadcasted(key, low, high):
    return genjax.uniform.sample(key, low, high)


def logpdf_uniform_broadcasted(values, low, high):
    valid = (low <= values) & (values <= high)
    position_score = jnp.log((valid * 1.0) * (jnp.ones_like(values) / (high - low)))
    return position_score.sum()


uniform_broadcasted = genjax.exact_density(
    sample_uniform_broadcasted, logpdf_uniform_broadcasted
)

uniform_discrete = genjax.exact_density(
    lambda key, vals: jax.random.choice(key, vals),
    lambda sampled_val, vals: jnp.log(1.0 / (vals.shape[0])),
)
uniform_pose = genjax.exact_density(sample_uniform_pose, logpdf_uniform_pose)
uniform_scale = genjax.exact_density(sample_uniform_scale, logpdf_uniform_scale)

vmf = genjax.exact_density(
    lambda key, mean, concentration: tfp.distributions.VonMisesFisher(
        mean, concentration
    ).sample(seed=key),
    lambda x, mean, concentration: tfp.distributions.VonMisesFisher(
        mean, concentration
    ).log_prob(x),
)

gaussian_vmf = genjax.exact_density(sample_gaussian_vmf_pose, logpdf_gaussian_vmf_pose)

### Below are placeholders for genjax functions which are currently buggy ###
## TODO: these bugs in genjax should now be fixed, so we should be able to
# remove these.

# There is currently a bug in `genjax.uniform.logpdf`; this `uniform`
# can be used instead until a fix is pushed.
uniform = genjax.exact_density(
    lambda key, low, high: genjax.uniform.sample(key, low, high),
    lambda x, low, high: jnp.sum(genjax.uniform.logpdf(x, low, high)),
)


def tfp_distribution(dist):
    def sampler(key, *args, **kwargs):
        d = dist(*args, **kwargs)
        return d.sample(seed=key)

    def logpdf(v, *args, **kwargs):
        d = dist(*args, **kwargs)
        return jnp.sum(d.log_prob(v))

    return genjax.exact_density(sampler, logpdf)


categorical = tfp_distribution(
    lambda logits: tfp.distributions.Categorical(logits=logits)
)
bernoulli = tfp_distribution(lambda logits: tfp.distributions.Bernoulli(logits=logits))
normal = tfp_distribution(tfp.distributions.Normal)


###


@Pytree.dataclass
class RenormalizedLaplace(genjax.ExactDensity):
    @jax.jit
    def sample(self, key, loc, scale, low, high):
        warnings.warn(
            "RenormalizedLaplace sampling is currently not implemented perfectly."
        )
        x = tfp.distributions.Laplace(loc, scale).sample(seed=key)
        return jnp.clip(x, low, high)

    @jax.jit
    def logpdf(self, obs, loc, scale, low, high):
        laplace_logpdf = tfp.distributions.Laplace(loc, scale).log_prob(obs)
        p_below_low = tfp.distributions.Laplace(loc, scale).cdf(low)
        p_below_high = tfp.distributions.Laplace(loc, scale).cdf(high)
        log_integral_of_laplace_pdf_within_this_range = jnp.log(
            p_below_high - p_below_low
        )
        logpdf_if_in_range = (
            laplace_logpdf - log_integral_of_laplace_pdf_within_this_range
        )

        return jnp.where(
            jnp.logical_and(obs >= low, obs <= high),
            logpdf_if_in_range,
            -jnp.inf,
        )


renormalized_laplace = RenormalizedLaplace()


@Pytree.dataclass
class RenormalizedColorLaplace(genjax.ExactDensity):
    @jax.jit
    def sample(self, key, loc, scale):
        return jax.vmap(
            lambda k, c: renormalized_laplace.sample(k, c, scale, 0.0, 1.0),
        )(jax.random.split(key, loc.shape[0]), loc)

    @jax.jit
    def logpdf(self, obs, loc, scale):
        return jax.vmap(
            lambda o, c: renormalized_laplace.logpdf(o, c, scale, 0.0, 1.0),
        )(obs, loc).sum()


renormalized_color_laplace = RenormalizedColorLaplace()

### Mixture distribution combinator ###


@Pytree.dataclass
class PythonMixtureDistribution(genjax.ExactDensity):
    """
    Mixture of different distributions.
    Constructor:
    - dists : python list of N genjax.ExactDensity objects

    Distribution args:
    - probs : (N,) array of branch probabilities
    - args : python list of argument tuples, so that
        `dists[i].sample(key, *args[i])` is valid for each i
    """

    dists: any = genjax.Pytree.static()

    def sample(self, key, probs, args):
        values = []
        for i, dist in enumerate(self.dists):
            key, subkey = jax.random.split(key)
            values.append(dist.sample(subkey, *args[i]))
        values = jnp.array(values)
        key, subkey = jax.random.split(key)
        component = genjax.categorical.sample(subkey, jnp.log(probs))
        return values[component]

    def logpdf(self, observed, probs, args):
        logprobs = []
        for i, dist in enumerate(self.dists):
            lp = dist.logpdf(observed, *args[i])
            logprobs.append(lp + jnp.log(probs[i]))
        logprobs = jnp.stack(logprobs)
        return jax.scipy.special.logsumexp(logprobs)


### Truncated laplace distribution, and mapped version for RGB ###


@Pytree.dataclass
class TruncatedLaplace(genjax.ExactDensity):
    """
    This is a distribution on the interval (low, high).
    The generative process is:
    1. Sample x ~ laplace(loc, scale).
    2. If x < low, sample y ~ uniform(low, low + uniform_window_size) and return y.
    3. If x > high, sample y ~ uniform(high - uniform_window_size, high) and return y.
    4. Otherwise, return x.

    Args:
    - loc: float
    - scale: float
    - low: float
    - high: float
    - uniform_window_size: float

    Support:
    - x in (low, high) [a float]
    """

    def sample(self, key, loc, scale, low, high, uniform_window_size):
        isvalid = jnp.logical_and(
            low < high, low + uniform_window_size < high - uniform_window_size
        )
        k1, k2 = jax.random.split(key, 2)
        x = tfp.distributions.Laplace(loc, scale).sample(seed=k1)
        u = jax.random.uniform(k2, ()) * uniform_window_size
        return jnp.where(
            isvalid,
            jnp.where(
                x > high, high - uniform_window_size + u, jnp.where(x < low, low + u, x)
            ),
            jnp.nan,
        )

    def logpdf(self, obs, loc, scale, low, high, uniform_window_size):
        isvalid = jnp.logical_and(
            low < high, low + uniform_window_size < high - uniform_window_size
        )
        laplace_logpdf = tfp.distributions.Laplace(loc, scale).log_prob(obs)
        laplace_logp_below_low = tfp.distributions.Laplace(loc, scale).log_cdf(low)
        laplace_logp_above_high = tfp.distributions.Laplace(
            loc, scale
        ).log_survival_function(high)
        log_window_size = jnp.log(uniform_window_size)

        score = jnp.where(
            jnp.logical_or(obs < low, obs > high),
            -jnp.inf,
            jnp.where(
                jnp.logical_and(
                    low + uniform_window_size < obs, obs < high - uniform_window_size
                ),
                laplace_logpdf,
                jnp.where(
                    obs < low + uniform_window_size,
                    jnp.logaddexp(
                        laplace_logp_below_low - log_window_size, laplace_logpdf
                    ),
                    jnp.logaddexp(
                        laplace_logp_above_high - log_window_size, laplace_logpdf
                    ),
                ),
            ),
        )
        return jnp.where(isvalid, score, -jnp.inf)


truncated_laplace = TruncatedLaplace()

_FIXED_COLOR_UNIFORM_WINDOW = 1 / 255
_FIXED_DEPTH_UNIFORM_WINDOW = 0.01


@Pytree.dataclass
class TruncatedColorLaplace(genjax.ExactDensity):
    """
    Args:
    - loc: (3,) array (loc for R, G, B channels)
    - shared_scale: float (scale, shared across R, G, B channels)
    - uniform_window_size: float [optional; defaults to 1/255]

    Support:
    - rgb in [0, 1]^3 [a 3D array]
    """

    def sample(
        self, key, loc, shared_scale, uniform_window_size=_FIXED_COLOR_UNIFORM_WINDOW
    ):
        return jax.vmap(
            lambda k, lc: truncated_laplace.sample(
                k, lc, shared_scale, 0.0, 1.0, uniform_window_size
            ),
            in_axes=(0, 0),
        )(jax.random.split(key, loc.shape[0]), loc)

    def logpdf(
        self, obs, loc, shared_scale, uniform_window_size=_FIXED_COLOR_UNIFORM_WINDOW
    ):
        return jax.vmap(
            lambda o, lc: truncated_laplace.logpdf(
                o, lc, shared_scale, 0.0, 1.0, uniform_window_size
            ),
            in_axes=(0, 0),
        )(obs, loc).sum()


truncated_color_laplace = TruncatedColorLaplace()
