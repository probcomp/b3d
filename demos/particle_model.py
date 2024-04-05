import genjax
import jax.numpy as jnp
import jax
import b3d

width=100
height=100
fx=50.0
fy=50.0
cx=50.0
cy=50.0
near=0.001
far=16.0
renderer = b3d.Renderer(
    width, height, fx, fy, cx, cy, near, far
)

@genjax.unfold_combinator(max_length=50)
@genjax.static_gen_fn
def unfolded_kernel(
    state, 
    num_particles_dummy,
    num_objects_dummy,                    
):
    num_particles = len(num_particles_dummy)
    num_objects = len(num_objects_dummy)

    (t, object_poses, particle_relative_poses, cluster_assignments, absolute_particle_poses) = state

    new_object_poses = genjax.map_combinator(in_axes=(0, None, None))(
        b3d.gaussian_vmf_pose
    )(object_poses, 0.1, 1000.0) @ "object_poses"

    new_particle_relative_poses = particle_relative_poses#if you want static
    new_particle_relative_poses = genjax.map_combinator(in_axes=(0, None, None))(
        b3d.gaussian_vmf_pose
    )(particle_relative_poses, 0.01, 1000.0) @ "particle_relative_poses"

    new_absolute_particle_poses = new_object_poses[cluster_assignments] @ new_particle_relative_poses

    return (t+1, new_object_poses, new_particle_relative_poses, cluster_assignments, new_absolute_particle_poses)


@genjax.static_gen_fn
def dhgps(
    num_particles_dummy,
    num_objects_dummy,
):
    num_particles = len(num_particles_dummy)
    num_objects = len(num_objects_dummy)
    
    object_poses = genjax.map_combinator(in_axes=(0,None,))(b3d.uniform_pose)(
        jnp.tile((-100.0 * jnp.ones(3))[None,...], (num_objects, 1)), jnp.ones(3) * 100.0
    ) @ "initial_object_pose" # (num_objects,)

    particle_relative_poses = genjax.map_combinator(in_axes=(0,None,))(b3d.uniform_pose)(
        jnp.tile((-100.0 * jnp.ones(3))[None,...], (num_particles, 1)), jnp.ones(3) * 100.0
    ) @ "initial_particle_relative_poses" # (num_particles,)

    object_assignments = genjax.map_combinator(in_axes=(0,))(genjax.categorical)(
        jnp.ones((num_particles, num_objects))
    ) @ "object_assignments" # (num_particles,)

    absolute_particle_poses = object_poses[object_assignments] @ particle_relative_poses

    state0 = (0, object_poses, particle_relative_poses, object_assignments, absolute_particle_poses)

    T = 10
    (_, object_poses, particle_relative_poses, object_assignments, absolute_particle_poses) =  unfolded_kernel(T, state0, num_particles_dummy, num_objects_dummy) @ "steps"
    return object_poses, particle_relative_poses, object_assignments, absolute_particle_poses

trace = dhgps.simulate(jax.random.PRNGKey(1101), (jnp.arange(100), jnp.arange(5), ))
object_poses = trace["object_pose",:]
particle_relative_poses = trace["particle_relative_poses",:]
object_assignments = trace["object_assignments",:]

trace.get_retval()[-1].shape

trace["object_assignments",:]


import rerun as rr
rr.init("demo.py")
rr.connect("127.0.0.1:8812")

absolute_particles = trace.get_retval()[0]
print(absolute_particles.shape)