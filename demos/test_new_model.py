import b3d
from b3d.model import model_multiobject_gl_factory
from b3d import Pose
import jax
import genjax
import jax.numpy as jnp


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

key = jax.random.PRNGKey(110)

object_library = b3d.MeshLibrary.make_empty_library()
object_library.add_object(jnp.zeros((100,3)), jnp.zeros((10,3),dtype=jnp.int32), jnp.zeros((100,3)))
object_library.add_object(jnp.zeros((100,3)), jnp.zeros((10,3),dtype=jnp.int32), jnp.zeros((100,3)))
object_library.add_object(jnp.zeros((100,3)), jnp.zeros((10,3),dtype=jnp.int32), jnp.zeros((100,3)))



@genjax.static_gen_fn
def type_and_pose_gf(_, object_library):
    object_identity = b3d.uniform_discrete(jnp.arange(-1, object_library.get_num_objects())) @ f"type"
    object_pose = b3d.uniform_pose(jnp.ones(3)*-100.0, jnp.ones(3)*100.0) @ f"pose"
    return object_identity, object_pose

trace = type_and_pose_gf.simulate(key, (None, object_library,))
print(trace["pose"])
print(trace["type"])

@genjax.static_gen_fn
def particle_properties_gf(_, dummy_num_objects):
    particle_pose = b3d.uniform_pose(jnp.ones(3)*-100.0, jnp.ones(3)*100.0) @ f"particle_pose"
    assignment =  b3d.uniform_discrete(jnp.arange(0, len(dummy_num_objects))) @ f"assignment"
    return particle_pose, assignment

trace = particle_properties_gf.simulate(key, (None, jnp.arange(5),))
print(trace["particle_pose"])
print(trace["assignment"])


@genjax.static_gen_fn
def scene_model(dummy_num_objects, dummy_num_particles, object_library):
    num_particles = len(dummy_num_particles)
    num_objects = len(dummy_num_objects)

    object_identities, object_poses = genjax.map_combinator(in_axes=(0, None,))(
        type_and_pose_gf
    )(dummy_num_objects, object_library) @ "objects"

    particle_poses, object_assignment = genjax.map_combinator(in_axes=(0, None,))(
        particle_properties_gf
    )(dummy_num_particles, dummy_num_objects) @ "properties"

    camera_pose = b3d.uniform_pose(jnp.ones(3)*-100.0, jnp.ones(3)*100.0) @ f"camera_pose"
    absolute_particle_poses = object_poses[object_assignment] @ particle_poses

    absolute_particle_poses_in_camera_frame = camera_pose.inv() @ absolute_particle_poses
    return absolute_particle_poses_in_camera_frame





trace = scene_model.simulate(jax.random.PRNGKey(1101), (jnp.arange(5), jnp.arange(1000), object_library,))

print(trace["objects", :, "type"])
print(trace["objects", :, "pose"])
print(trace["properties", :, "particle_pose"])
print(trace["properties", :, "assignment"])






addr_list = ("objects", 0, "pose")
value = Pose.identity()[None,...]

choice_map = None
for i in range(len(addr_list)-1, -1, -1):
    if choice_map is None:
        choice_map = genjax.choice_map({addr_list[i]: value})
    elif isinstance(addr_list[i], int):
        choice_map = genjax.indexed_choice_map(jnp.array([addr_list[i]]), choice_map)
    else:
        choice_map = genjax.choice_map({addr_list[i]: choice_map})

trace = trace.update(
    key,
    choice_map,
    genjax.Diff.tree_diff_unknown_change(trace.get_args())
)[0]
print(trace["objects", 1, "pose"])
print(trace["objects", 0, "pose"])

def make_hierarchical_choice_map(address_list, value):
    choice_map = None
    for i in range(len(address_list)-1, -1, -1):
        if i == len(address_list)-1:
            choice_map = genjax.choice_map({address_list[i]: value})
        elif isinstance(address_list[i], int):
            choice_map = genjax.indexed_choice_map(jnp.array([address_list[i]]), choice_map)
        else:
            choice_map = genjax.choice_map({address_list[i]: choice_map})
    return choice_map

[("object", 1, "pose"), ("object", 2, "particle_pose")]



def update_choice(trace, key, address, value):
    address_list = address.const
    choice_map = None

    for i in range(len(addr_list)-1, -1, -1):
        if choice_map is None:
            choice_map = genjax.choice_map({addr_list[i]: value})
        elif isinstance(addr_list[i], int):
            choice_map = genjax.indexed_choice_map(jnp.array([addr_list[i]]), choice_map)
        else:
            choice_map = genjax.choice_map({addr_list[i]: choice_map})


    return trace.update(
        key,
        choice_map,
        genjax.Diff.tree_diff_unknown_change(trace.get_args())
    )[0]
update_choices_jit = jax.jit(update_choices)

trace = update_choices_jit(
    trace, key,
    genjax.Pytree.const(("objects", 0, "pose")),
    Pose.identity()[None,...]
)