

N = 1000
def accumulate(carry, x):
    (key, pose) = carry
    keys = jax.random.split(key, N)
    test_poses = jax.vmap(Pose.sample_gaussian_vmf_pose, in_axes=(0, None, None, None))(keys, pose, 0.01, 500.0)
    scores = b3d.enumerate_choices_get_scores_jit(gt_trace, key, genjax.Pytree.const(["object_pose_0"]), test_poses)
    key = jax.random.split(keys[-1],2)[0]
    # pose = test_poses[jax.random.categorical(key, scores)]
    pose = test_poses[scores.argmax()]
    return (key, pose), pose
f = jax.jit(lambda arg1, arg2: jax.lax.scan(accumulate, arg1, arg2))

traces = []
for _ in range(10):
    key = jax.random.split(key, 1)[0]
    pose =  Pose.sample_gaussian_vmf_pose(key, object_pose, 0.01, 1.0)
    trace = b3d.update_choices_jit(gt_trace, key, genjax.Pytree.const(["object_pose_0"]), pose)
    print(trace.get_score())
    (key,pose), _ = f((key, pose), jnp.arange(50))
    trace = b3d.update_choices_jit(gt_trace, key, genjax.Pytree.const(["object_pose_0"]), pose)
    print(trace.get_score())
    traces.append(trace)
    b3d.rerun_visualize_trace_t(trace, 0)

for i in range(len(traces)):
    b3d.rerun_visualize_trace_t(traces[i], i)

