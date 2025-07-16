import warp as wp
from warp.jax_experimental.ffi import jax_callable
from warp.sim.collide import closest_point_plane, counter_increment
import jax
from functools import partial


@wp.kernel(enable_backward=False)
def broadphase_collision_pairs(
    contact_pairs: wp.array(dtype=int, ndim=2),
    body_q: wp.array(dtype=wp.transform),
    shape_X_bs: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    body_mass: wp.array(dtype=float),
    geo_type: wp.array(dtype=wp.int32),
    geo_scale: wp.array(dtype=wp.vec3),
    geo_source: wp.array(dtype=wp.uint64),
    collision_radius: wp.array(dtype=float),
    rigid_contact_max: int,
    rigid_contact_margin: float,
    # outputs
    contact_count: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_point_id: wp.array(dtype=int),
):
    tid = wp.tid()
    shape_a = contact_pairs[tid, 0]
    shape_b = contact_pairs[tid, 1]

    mass_a = 0.0
    mass_b = 0.0
    rigid_a = shape_body[shape_a]
    if rigid_a == -1:
        X_ws_a = shape_X_bs[shape_a]
    else:
        X_ws_a = wp.transform_multiply(body_q[rigid_a], shape_X_bs[shape_a])
        mass_a = body_mass[rigid_a]
    rigid_b = shape_body[shape_b]
    if rigid_b == -1:
        X_ws_b = shape_X_bs[shape_b]
    else:
        X_ws_b = wp.transform_multiply(body_q[rigid_b], shape_X_bs[shape_b])
        mass_b = body_mass[rigid_b]
    if mass_a == 0.0 and mass_b == 0.0:
        # skip if both bodies are static
        return

    type_a = geo_type[shape_a]
    type_b = geo_type[shape_b]
    # unique ordering of shape pairs
    if type_a < type_b:
        actual_shape_a = shape_a
        actual_shape_b = shape_b
        actual_type_a = type_a
        actual_type_b = type_b
        actual_X_ws_a = X_ws_a
        actual_X_ws_b = X_ws_b
    else:
        actual_shape_a = shape_b
        actual_shape_b = shape_a
        actual_type_a = type_b
        actual_type_b = type_a
        actual_X_ws_a = X_ws_b
        actual_X_ws_b = X_ws_a

    p_a = wp.transform_get_translation(actual_X_ws_a)
    if actual_type_b == wp.sim.GEO_PLANE:
        if actual_type_a == wp.sim.GEO_PLANE:
            return
        query_b = wp.transform_point(wp.transform_inverse(actual_X_ws_b), p_a)
        scale = geo_scale[actual_shape_b]
        closest = closest_point_plane(scale[0], scale[1], query_b)
        d = wp.length(query_b - closest)
        r_a = collision_radius[actual_shape_a]
        if d > r_a + rigid_contact_margin:
            return
    else:
        p_b = wp.transform_get_translation(actual_X_ws_b)
        d = wp.length(p_a - p_b) * 0.5 - 0.1
        r_a = collision_radius[actual_shape_a]
        r_b = collision_radius[actual_shape_b]
        if d > r_a + r_b + rigid_contact_margin:
            return

    # determine how many contact points need to be evaluated
    num_contacts = 0
    if actual_type_a == wp.sim.GEO_MESH:
        mesh_a = wp.mesh_get(geo_source[actual_shape_a])
        num_contacts_a = mesh_a.points.shape[0]
        num_contacts_b = 0
        if actual_type_b == wp.sim.GEO_MESH:
            mesh_b = wp.mesh_get(geo_source[actual_shape_b])
            num_contacts_b = mesh_b.points.shape[0]
        elif actual_type_b != wp.sim.GEO_PLANE:
            print("broadphase_collision_pairs: unsupported geometry type for mesh collision")
            return
        num_contacts = num_contacts_a + num_contacts_b
        # wp.print("num_contacts_a")
        # wp.print(num_contacts_a)
        # wp.print("num_contacts_b")
        # wp.print(num_contacts_b)
        if num_contacts > 0:
            # wp.print("num_contacts")
            # wp.print(num_contacts)
            index = wp.atomic_add(contact_count, 0, num_contacts)
            # wp.print("index")
            # wp.print(index)
            # wp.print("rigid_contact_max")
            # wp.print(rigid_contact_max)
            if index + num_contacts - 1 >= rigid_contact_max:
                print("Mesh contact: Number of rigid contacts exceeded limit. Increase Model.rigid_contact_max.")
                return
            # allocate contact points from mesh A against B
            for i in range(num_contacts_a):
                contact_shape0[index + i] = actual_shape_a
                contact_shape1[index + i] = actual_shape_b
                contact_point_id[index + i] = i
            # allocate contact points from mesh B against A
            for i in range(num_contacts_b):
                contact_shape0[index + num_contacts_a + i] = actual_shape_b
                contact_shape1[index + num_contacts_a + i] = actual_shape_a
                contact_point_id[index + num_contacts_a + i] = i

        return
    
@wp.kernel
def handle_contact_pairs(
    body_q: wp.array(dtype=wp.transform),
    shape_X_bs: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    geo_type: wp.array(dtype=wp.int32),
    geo_scale: wp.array(dtype=wp.vec3),
    geo_source: wp.array(dtype=wp.uint64),
    geo_thickness: wp.array(dtype=float),
    rigid_contact_margin: float,
    contact_broad_shape0: wp.array(dtype=int),
    contact_broad_shape1: wp.array(dtype=int),
    contact_point_id: wp.array(dtype=int),
    # outputs
    contact_count: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_offset0: wp.array(dtype=wp.vec3),
    contact_offset1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_thickness: wp.array(dtype=float),
    contact_tids: wp.array(dtype=int),
):
    tid = wp.tid()
    shape_a = contact_broad_shape0[tid]
    shape_b = contact_broad_shape1[tid]
    if shape_a == shape_b:
        return

    point_id = contact_point_id[tid]

    rigid_a = shape_body[shape_a]
    X_wb_a = wp.transform_identity()
    if rigid_a >= 0:
        X_wb_a = body_q[rigid_a]
    X_bs_a = shape_X_bs[shape_a]
    X_ws_a = wp.transform_multiply(X_wb_a, X_bs_a)
    X_bw_a = wp.transform_inverse(X_wb_a)
    geo_type_a = geo_type[shape_a]
    geo_scale_a = geo_scale[shape_a]
    min_scale_a = min(geo_scale_a)
    thickness_a = geo_thickness[shape_a]
    # is_solid_a = geo.is_solid[shape_a]

    rigid_b = shape_body[shape_b]
    X_wb_b = wp.transform_identity()
    if rigid_b >= 0:
        X_wb_b = body_q[rigid_b]
    X_bs_b = shape_X_bs[shape_b]
    X_ws_b = wp.transform_multiply(X_wb_b, X_bs_b)
    X_sw_b = wp.transform_inverse(X_ws_b)
    X_bw_b = wp.transform_inverse(X_wb_b)
    geo_type_b = geo_type[shape_b]
    geo_scale_b = geo_scale[shape_b]
    min_scale_b = min(geo_scale_b)
    thickness_b = geo_thickness[shape_b]
    # is_solid_b = geo.is_solid[shape_b]

    distance = 1.0e6
    thickness = thickness_a + thickness_b

    if geo_type_a == wp.sim.GEO_MESH and geo_type_b == wp.sim.GEO_MESH:
        # vertex-based contact
        mesh = wp.mesh_get(geo_source[shape_a])
        mesh_b = geo_source[shape_b]

        body_a_pos = wp.cw_mul(mesh.points[point_id], geo_scale_a)
        p_a_world = wp.transform_point(X_ws_a, body_a_pos)
        query_b_local = wp.transform_point(X_sw_b, p_a_world)

        face_index = int(0)
        face_u = float(0.0)
        face_v = float(0.0)
        sign = float(0.0)
        min_scale = min(min_scale_a, min_scale_b)
        max_dist = (rigid_contact_margin + thickness) / min_scale

        res = wp.mesh_query_point_sign_normal(
            mesh_b, wp.cw_div(query_b_local, geo_scale_b), max_dist, sign, face_index, face_u, face_v
        )

        if res:
            shape_p = wp.mesh_eval_position(mesh_b, face_index, face_u, face_v)
            shape_p = wp.cw_mul(shape_p, geo_scale_b)
            p_b_world = wp.transform_point(X_ws_b, shape_p)
            # contact direction vector in world frame
            diff_b = p_a_world - p_b_world
            normal = wp.normalize(diff_b) * sign
            distance = wp.dot(diff_b, normal)
        else:
            return

    elif geo_type_a == wp.sim.GEO_MESH and geo_type_b == wp.sim.GEO_PLANE:
        # vertex-based contact
        mesh = wp.mesh_get(geo_source[shape_a])
        body_a_pos = wp.cw_mul(mesh.points[point_id], geo_scale_a)
        p_a_world = wp.transform_point(X_ws_a, body_a_pos)
        query_b = wp.transform_point(X_sw_b, p_a_world)
        p_b_body = closest_point_plane(geo_scale_b[0], geo_scale_b[1], query_b)
        p_b_world = wp.transform_point(X_ws_b, p_b_body)
        diff = p_a_world - p_b_world

        # if the plane is infinite or the point is within the plane we fix the normal to prevent intersections
        if (
            geo_scale_b[0] == 0.0
            and geo_scale_b[1] == 0.0
            or wp.abs(query_b[0]) < geo_scale_b[0]
            and wp.abs(query_b[2]) < geo_scale_b[1]
        ):
            normal = wp.transform_vector(X_ws_b, wp.vec3(0.0, 1.0, 0.0))
            distance = wp.dot(diff, normal)
        else:
            normal = wp.normalize(diff)
            distance = wp.dot(diff, normal)
            # ignore extreme penetrations (e.g. when mesh is below the plane)
            if distance < -rigid_contact_margin:
                return

    else:
        wp.print("geo_type_a")
        wp.print(geo_type_a)
        wp.print("geo_type_b")
        wp.print(geo_type_b)
        print("Unsupported geometry pair in collision handling")
        return

    d = distance - thickness
    if d < rigid_contact_margin:
        index = counter_increment(contact_count, 0, contact_tids, tid)
        contact_shape0[index] = shape_a
        contact_shape1[index] = shape_b
        # transform from world into body frame (so the contact point includes the shape transform)
        contact_point0[index] = wp.transform_point(X_bw_a, p_a_world)
        contact_point1[index] = wp.transform_point(X_bw_b, p_b_world)
        contact_offset0[index] = wp.transform_vector(X_bw_a, -thickness_a * normal)
        contact_offset1[index] = wp.transform_vector(X_bw_b, thickness_b * normal)
        contact_normal[index] = normal
        contact_thickness[index] = thickness

@wp.kernel
def eval_rigid_contacts(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    shape_ke: wp.array(dtype=float),
    shape_kd: wp.array(dtype=float),
    shape_kf: wp.array(dtype=float),
    shape_ka: wp.array(dtype=float),
    shape_mu: wp.array(dtype=float),
    geo_thickness: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    force_in_world_frame: bool,
    friction_smoothing: float,
    noise: bool,
    # outputs
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    count = contact_count[0]
    if tid >= count:
        return

    # retrieve contact thickness, compute average contact material properties
    ke = 0.0  # contact normal force stiffness
    kd = 0.0  # damping coefficient
    kf = 0.0  # friction force stiffness
    ka = 0.0  # adhesion distance
    mu = 0.0  # friction coefficient
    mat_nonzero = 0
    thickness_a = 0.0
    thickness_b = 0.0
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        return
    body_a = -1
    body_b = -1
    if shape_a >= 0:
        mat_nonzero += 1
        ke += shape_ke[shape_a]
        kd += shape_kd[shape_a]
        kf += shape_kf[shape_a]
        ka += shape_ka[shape_a]
        mu += shape_mu[shape_a]
        thickness_a = geo_thickness[shape_a]
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        mat_nonzero += 1
        ke += shape_ke[shape_b]
        kd += shape_kd[shape_b]
        kf += shape_kf[shape_b]
        ka += shape_ka[shape_b]
        mu += shape_mu[shape_b]
        thickness_b = geo_thickness[shape_b]
        body_b = shape_body[shape_b]
    if mat_nonzero > 0:
        ke /= float(mat_nonzero)
        kd /= float(mat_nonzero)
        kf /= float(mat_nonzero)
        ka /= float(mat_nonzero)
        mu /= float(mat_nonzero)

    # contact normal in world space
    n = contact_normal[tid]
    bx_a = contact_point0[tid]
    bx_b = contact_point1[tid]
    r_a = wp.vec3(0.0)
    r_b = wp.vec3(0.0)
    if body_a >= 0:
        X_wb_a = body_q[body_a]
        X_com_a = body_com[body_a]
        bx_a = wp.transform_point(X_wb_a, bx_a) - thickness_a * n
        r_a = bx_a - wp.transform_point(X_wb_a, X_com_a)

    if body_b >= 0:
        X_wb_b = body_q[body_b]
        X_com_b = body_com[body_b]
        bx_b = wp.transform_point(X_wb_b, bx_b) + thickness_b * n
        r_b = bx_b - wp.transform_point(X_wb_b, X_com_b)

    d = wp.dot(n, bx_a - bx_b)

    if d >= ka:
        return

    # compute contact point velocity
    bv_a = wp.vec3(0.0)
    bv_b = wp.vec3(0.0)
    if body_a >= 0:
        body_v_s_a = body_qd[body_a]
        body_w_a = wp.spatial_top(body_v_s_a)
        body_v_a = wp.spatial_bottom(body_v_s_a)
        if force_in_world_frame:
            bv_a = body_v_a + wp.cross(body_w_a, bx_a)
        else:
            bv_a = body_v_a + wp.cross(body_w_a, r_a)

    if body_b >= 0:
        body_v_s_b = body_qd[body_b]
        body_w_b = wp.spatial_top(body_v_s_b)
        body_v_b = wp.spatial_bottom(body_v_s_b)
        if force_in_world_frame:
            bv_b = body_v_b + wp.cross(body_w_b, bx_b)
        else:
            bv_b = body_v_b + wp.cross(body_w_b, r_b)

    # relative velocity
    v = bv_a - bv_b

    # print(v)

    # decompose relative velocity
    vn = wp.dot(n, v)
    vt = v - n * vn

    # contact elastic
    fn = d * ke
    if noise:
        r = wp.rand_init(tid)
        fn = fn + 1000000. * wp.randn(r)

    # contact damping
    fd = wp.min(vn, 0.0) * kd * wp.step(d)

    # viscous friction
    # ft = vt*kf

    # Coulomb friction (box)
    # lower = mu * d * ke
    # upper = -lower

    # vx = wp.clamp(wp.dot(wp.vec3(kf, 0.0, 0.0), vt), lower, upper)
    # vz = wp.clamp(wp.dot(wp.vec3(0.0, 0.0, kf), vt), lower, upper)

    # ft = wp.vec3(vx, 0.0, vz)

    # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    ft = wp.vec3(0.0)
    if d < 0.0:
        # use a smooth vector norm to avoid gradient instability at/around zero velocity
        vs = wp.norm_huber(vt, delta=friction_smoothing)
        if vs > 0.0:
            fr = vt / vs
            ft = fr * wp.min(kf * vs, -mu * (fn + fd))

    if noise:
        r = wp.rand_init(tid)
        n_x = n[0] + 10. * wp.randn(r)
        r = wp.rand_init(tid+1)
        n_y = n[1] + 10. * wp.randn(r)
        r = wp.rand_init(tid+2)
        n_z = n[2] + 10. * wp.randn(r)
        n = wp.vec3(n_x, n_y, n_z)
        n = wp.normalize(n)

    f_total = n * (fn + fd) + ft
    # f_total = n * (fn + fd)
    # f_total = n * fn

    if body_a >= 0:
        if force_in_world_frame:
            wp.atomic_add(body_f, body_a, wp.spatial_vector(wp.cross(bx_a, f_total), f_total))
        else:
            wp.atomic_sub(body_f, body_a, wp.spatial_vector(wp.cross(r_a, f_total), f_total))

    if body_b >= 0:
        if force_in_world_frame:
            wp.atomic_sub(body_f, body_b, wp.spatial_vector(wp.cross(bx_b, f_total), f_total))
        else:
            wp.atomic_add(body_f, body_b, wp.spatial_vector(wp.cross(r_b, f_total), f_total))

@wp.func
def integrate_rigid_body(
    q: wp.transform,
    qd: wp.spatial_vector,
    f: wp.spatial_vector,
    com: wp.vec3,
    inertia: wp.mat33,
    inv_mass: float,
    inv_inertia: wp.mat33,
    gravity: wp.vec3,
    angular_damping: float,
    dt: float,
):
    # unpack transform
    x0 = wp.transform_get_translation(q)
    r0 = wp.transform_get_rotation(q)

    # unpack spatial twist
    w0 = wp.spatial_top(qd)
    v0 = wp.spatial_bottom(qd)

    # unpack spatial wrench
    t0 = wp.spatial_top(f)
    f0 = wp.spatial_bottom(f)

    x_com = x0 + wp.quat_rotate(r0, com)

    # linear part
    v1 = v0 + (f0 * inv_mass + gravity * wp.nonzero(inv_mass)) * dt
    x1 = x_com + v1 * dt

    # angular part (compute in body frame)
    wb = wp.quat_rotate_inv(r0, w0)
    tb = wp.quat_rotate_inv(r0, t0) - wp.cross(wb, inertia * wb)  # coriolis forces

    w1 = wp.quat_rotate(r0, wb + inv_inertia * tb * dt)
    r1 = wp.normalize(r0 + wp.quat(w1, 0.0) * r0 * 0.5 * dt)

    # angular damping
    w1 *= 1.0 - angular_damping * dt

    q_new = wp.transform(x1 - wp.quat_rotate(r1, com), r1)
    qd_new = wp.spatial_vector(w1, v1)

    return q_new, qd_new

# semi-implicit Euler integration
@wp.kernel
def integrate_bodies(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    I: wp.array(dtype=wp.mat33),
    inv_m: wp.array(dtype=float),
    inv_I: wp.array(dtype=wp.mat33),
    g: float,
    angular_damping: float,
    dt: float,
    # outputs
    body_q_new: wp.array(dtype=wp.transform),
    body_qd_new: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    # positions
    q = body_q[tid]
    qd = body_qd[tid]
    f = body_f[tid]

    # masses
    inv_mass = inv_m[tid]  # 1 / mass

    inertia = I[tid]
    inv_inertia = inv_I[tid]  # inverse of 3x3 inertia matrix

    com = body_com[tid]

    q_new, qd_new = integrate_rigid_body(
        q,
        qd,
        f,
        com,
        inertia,
        inv_mass,
        inv_inertia,
        wp.vec3(0., g, 0.),
        angular_damping,
        dt,
    )

    body_q_new[tid] = q_new
    body_qd_new[tid] = qd_new


def collide(
    # inputs
    shape_contact_pair_count: int,
    shape_contact_pairs: wp.array(dtype=int, ndim=2),
    body_q: wp.array(dtype=wp.transform),
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    body_mass: wp.array(dtype=float),
    geo_type: wp.array(dtype=wp.int32),
    geo_scale: wp.array(dtype=wp.vec3),
    geo_source: wp.array(dtype=wp.uint64),
    geo_thickness: wp.array(dtype=float),
    shape_collision_radius: wp.array(dtype=float),
    rigid_contact_max: int,
    rigid_contact_margin: float,
    shape_ground_contact_pair_count: int,
    shape_ground_contact_pairs: wp.array(dtype=int, ndim=2),

    rigid_contact_count_in: wp.array(dtype=int),
    rigid_contact_broad_shape0_in: wp.array(dtype=int),
    rigid_contact_broad_shape1_in: wp.array(dtype=int),
    rigid_contact_point_id_in: wp.array(dtype=int),
    rigid_contact_shape0_in: wp.array(dtype=int),
    rigid_contact_shape1_in: wp.array(dtype=int),
    rigid_contact_point0_in: wp.array(dtype=wp.vec3),
    rigid_contact_point1_in: wp.array(dtype=wp.vec3),
    rigid_contact_offset0_in: wp.array(dtype=wp.vec3),
    rigid_contact_offset1_in: wp.array(dtype=wp.vec3),
    rigid_contact_normal_in: wp.array(dtype=wp.vec3),
    rigid_contact_thickness_in: wp.array(dtype=float),
    rigid_contact_tids_in: wp.array(dtype=int),

    # outputs (all are model's attributes)
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_broad_shape0: wp.array(dtype=int),
    rigid_contact_broad_shape1: wp.array(dtype=int),
    rigid_contact_point_id: wp.array(dtype=int),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_offset0: wp.array(dtype=wp.vec3),
    rigid_contact_offset1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    rigid_contact_thickness: wp.array(dtype=float),
    rigid_contact_tids: wp.array(dtype=int),
):
    
    wp.copy(rigid_contact_count, rigid_contact_count_in)
    wp.copy(rigid_contact_broad_shape0, rigid_contact_broad_shape0_in)
    wp.copy(rigid_contact_broad_shape1, rigid_contact_broad_shape1_in)
    wp.copy(rigid_contact_point_id, rigid_contact_point_id_in)
    wp.copy(rigid_contact_shape0, rigid_contact_shape0_in)
    wp.copy(rigid_contact_shape1, rigid_contact_shape1_in)
    wp.copy(rigid_contact_point0, rigid_contact_point0_in)
    wp.copy(rigid_contact_point1, rigid_contact_point1_in)
    wp.copy(rigid_contact_offset0, rigid_contact_offset0_in)
    wp.copy(rigid_contact_offset1, rigid_contact_offset1_in)
    wp.copy(rigid_contact_normal, rigid_contact_normal_in)
    wp.copy(rigid_contact_thickness, rigid_contact_thickness_in)
    wp.copy(rigid_contact_tids, rigid_contact_tids_in)

    rigid_contact_count.zero_()
    rigid_contact_broad_shape0.fill_(-1)
    rigid_contact_broad_shape1.fill_(-1)

    wp.launch(
        kernel=broadphase_collision_pairs,
        dim=shape_contact_pair_count,
        inputs=[
            shape_contact_pairs,
            body_q,
            shape_transform,
            shape_body,
            body_mass,
            geo_type,
            geo_scale,
            geo_source,
            shape_collision_radius,
            rigid_contact_max,
            rigid_contact_margin,
        ],
        outputs=[
            rigid_contact_count,
            rigid_contact_broad_shape0,
            rigid_contact_broad_shape1,
            rigid_contact_point_id,
        ],
        record_tape=False,
    )

    wp.launch(
        kernel=broadphase_collision_pairs,
        dim=shape_ground_contact_pair_count,
        inputs=[
            shape_ground_contact_pairs,
            body_q,
            shape_transform,
            shape_body,
            body_mass,
            geo_type,
            geo_scale,
            geo_source,
            shape_collision_radius,
            rigid_contact_max,
            rigid_contact_margin,
        ],
        outputs=[
            rigid_contact_count,
            rigid_contact_broad_shape0,
            rigid_contact_broad_shape1,
            rigid_contact_point_id,
        ],
        record_tape=False,
    )

    rigid_contact_count.zero_()
    rigid_contact_tids.zero_()
    rigid_contact_shape0.fill_(-1)
    rigid_contact_shape1.fill_(-1)

    wp.launch(
        kernel=handle_contact_pairs,
        dim=rigid_contact_max,
        inputs=[
            body_q,
            shape_transform,
            shape_body,
            geo_type,
            geo_scale,
            geo_source,
            geo_thickness,
            rigid_contact_margin,
            rigid_contact_broad_shape0,
            rigid_contact_broad_shape1,
            rigid_contact_point_id,
        ],
        outputs=[
            rigid_contact_count,
            rigid_contact_shape0,
            rigid_contact_shape1,
            rigid_contact_point0,
            rigid_contact_point1,
            rigid_contact_offset0,
            rigid_contact_offset1,
            rigid_contact_normal,
            rigid_contact_thickness,
            rigid_contact_tids,
        ],
    )


def simulate(
    # inputs
    rigid_contact_max: int,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    ke: wp.array(dtype=float),
    kd: wp.array(dtype=float),
    kf: wp.array(dtype=float),
    ka: wp.array(dtype=float),
    mu: wp.array(dtype=float),
    geo_thickness: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    rigid_contact_count: wp.array(dtype=int),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    rigid_contact_shape0: wp.array(dtype=int),
    rigid_contact_shape1: wp.array(dtype=int),
    body_count: int,
    body_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    g: float,
    dt: float,

    body_q_in: wp.array(dtype=wp.transform),
    body_qd_in: wp.array(dtype=wp.spatial_vector),
    body_f_in: wp.array(dtype=wp.spatial_vector),

    # outputs (all are state features)
    body_q_new: wp.array(dtype=wp.transform),
    body_qd_new: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    
    wp.copy(body_q_new, body_q_in)
    wp.copy(body_qd_new, body_qd_in)
    wp.copy(body_f, body_f_in)

    # compute forces
    wp.launch(
        kernel=eval_rigid_contacts,
        dim=rigid_contact_max,
        inputs=[
            body_q,
            body_qd,
            body_com,
            ke,
            kd,
            kf,
            ka,
            mu,
            geo_thickness,
            shape_body,
            rigid_contact_count,
            rigid_contact_point0,
            rigid_contact_point1,
            rigid_contact_normal,
            rigid_contact_shape0,
            rigid_contact_shape1,
            False,
            1.0,
            False,
        ],
        outputs=[body_f],
    )
    
    # integrate
    wp.launch(
        kernel=integrate_bodies,
        dim=body_count,
        inputs=[
            body_q,
            body_qd,
            body_f,
            body_com,
            body_inertia,
            body_inv_mass,
            body_inv_inertia,
            g,
            0.05,
            dt,
        ],
        outputs=[body_q_new, body_qd_new],
    )

jax_collide = jax_callable(collide, num_outputs=13)
jax_simulate = jax_callable(simulate, num_outputs=3)

@partial(jax.jit, static_argnames=["rigid_contact_max", "shape_contact_pair_count", "shape_ground_contact_pair_count", "rigid_contact_margin", "body_count", "g", "dt"])
def warp_step(rigid_contact_count, rigid_contact_broad_shape0, rigid_contact_broad_shape1, rigid_contact_point_id, rigid_contact_shape0, rigid_contact_shape1, rigid_contact_point0, rigid_contact_point1, rigid_contact_offset0, rigid_contact_offset1, rigid_contact_normal, rigid_contact_thickness, rigid_contact_tids, body_qd, body_f, body_com, ke, kd, kf, ka, mu, body_inertia, body_inv_mass, body_inv_inertia, shape_contact_pair_count, shape_contact_pairs, body_q, shape_transform, shape_body, body_mass, geo_type, geo_scale, geo_source, geo_thickness, shape_collision_radius, rigid_contact_max, rigid_contact_margin, shape_ground_contact_pair_count, shape_ground_contact_pairs, body_count, g, dt):
    
    # compute all collision info, output are all model features)
    output_dims_coll = {"rigid_contact_count": 1, "rigid_contact_broad_shape0": rigid_contact_max, "rigid_contact_broad_shape1": rigid_contact_max, "rigid_contact_point_id": rigid_contact_max, "rigid_contact_shape0": rigid_contact_max, "rigid_contact_shape1": rigid_contact_max, "rigid_contact_point0": rigid_contact_max, "rigid_contact_point1": rigid_contact_max, "rigid_contact_offset0": rigid_contact_max, "rigid_contact_offset1": rigid_contact_max, "rigid_contact_normal": rigid_contact_max, "rigid_contact_thickness": rigid_contact_max, "rigid_contact_tids": rigid_contact_max}
    rigid_contact_count, rigid_contact_broad_shape0, rigid_contact_broad_shape1, rigid_contact_point_id, rigid_contact_shape0, rigid_contact_shape1, rigid_contact_point0, rigid_contact_point1, rigid_contact_offset0, rigid_contact_offset1, rigid_contact_normal, rigid_contact_thickness, rigid_contact_tids = jax_collide(shape_contact_pair_count, shape_contact_pairs, body_q, shape_transform, shape_body, body_mass, geo_type, geo_scale, geo_source, geo_thickness, shape_collision_radius, rigid_contact_max, rigid_contact_margin, shape_ground_contact_pair_count, shape_ground_contact_pairs, rigid_contact_count, rigid_contact_broad_shape0, rigid_contact_broad_shape1, rigid_contact_point_id, rigid_contact_shape0, rigid_contact_shape1, rigid_contact_point0, rigid_contact_point1, rigid_contact_offset0, rigid_contact_offset1, rigid_contact_normal, rigid_contact_thickness, rigid_contact_tids, output_dims=output_dims_coll)

    # update body info, outputs are all state features)
    output_dims_sim = {"body_q_new": body_count, "body_qd_new": body_count, "body_f": body_count}
    body_q_new, body_qd_new, body_f_new = jax_simulate(rigid_contact_max, body_q, body_qd, body_com, ke, kd, kf, ka, mu, geo_thickness, shape_body, rigid_contact_count, rigid_contact_point0, rigid_contact_point1, rigid_contact_normal, rigid_contact_shape0, rigid_contact_shape1, body_count, body_inertia, body_inv_mass, body_inv_inertia, g, dt, body_q, body_qd, body_f, output_dims=output_dims_sim)

    return body_q_new, body_qd_new, body_f_new, rigid_contact_count, rigid_contact_broad_shape0, rigid_contact_broad_shape1, rigid_contact_point_id, rigid_contact_shape0, rigid_contact_shape1, rigid_contact_point0, rigid_contact_point1, rigid_contact_offset0, rigid_contact_offset1, rigid_contact_normal, rigid_contact_thickness, rigid_contact_tids

def step(model, state, hyperparams):
    state.from_pos_quat()
    print(f"body_q inside step: {state._body_q}")
    print(f"body_qd inside step: {state._body_qd}")
    print(f"body_f inside step: {state._body_f}")
    for _ in range(hyperparams["sim_substeps"].unwrap()):
        state.clear_forces()

        ## extract the current model and state parameters
        model_attributes = vars(model)
        rigid_contact_count, rigid_contact_broad_shape0, rigid_contact_broad_shape1, shape_contact_pairs, shape_transform, shape_body, body_mass, geo_type, geo_scale, geo_source, geo_thickness, shape_collision_radius, rigid_contact_point_id, shape_ground_contact_pairs, rigid_contact_tids, rigid_contact_shape0, rigid_contact_shape1, rigid_contact_point0, rigid_contact_point1, rigid_contact_offset0, rigid_contact_offset1, rigid_contact_normal, rigid_contact_thickness, body_com, body_inertia, body_inv_mass, body_inv_inertia, ke, kd, kf, ka, mu = model_attributes.values()
        state_attributes = vars(state)
        body_q, body_qd, body_f = state_attributes.values()
        
        body_q_new, body_qd_new, body_f_new, rigid_contact_count_new, rigid_contact_broad_shape0_new, rigid_contact_broad_shape1_new, rigid_contact_point_id_new, rigid_contact_shape0_new, rigid_contact_shape1_new, rigid_contact_point0_new, rigid_contact_point1_new, rigid_contact_offset0_new, rigid_contact_offset1_new, rigid_contact_normal_new, rigid_contact_thickness_new, rigid_contact_tids_new = warp_step(rigid_contact_count, rigid_contact_broad_shape0, rigid_contact_broad_shape1, rigid_contact_point_id, rigid_contact_shape0, rigid_contact_shape1, rigid_contact_point0, rigid_contact_point1, rigid_contact_offset0, rigid_contact_offset1, rigid_contact_normal, rigid_contact_thickness, rigid_contact_tids, body_qd, body_f, body_com, ke, kd, kf, ka, mu, body_inertia, body_inv_mass, body_inv_inertia, hyperparams['shape_contact_pair_count'].unwrap(), shape_contact_pairs, body_q, shape_transform, shape_body, body_mass, geo_type, geo_scale, geo_source, geo_thickness, shape_collision_radius, hyperparams["rigid_contact_max"].unwrap(), hyperparams['rigid_contact_margin'].unwrap(), hyperparams['shape_ground_contact_pair_count'].unwrap(), shape_ground_contact_pairs, hyperparams['body_count'].unwrap(), hyperparams['g'].unwrap(), hyperparams['sim_dt'].unwrap())

        ## update the model and state parameters
        model.update_attributes(_rigid_contact_count = rigid_contact_count_new, _rigid_contact_broad_shape0 = rigid_contact_broad_shape0_new, _rigid_contact_broad_shape1 = rigid_contact_broad_shape1_new, _rigid_contact_point_id = rigid_contact_point_id_new, _rigid_contact_shape0 = rigid_contact_shape0_new, _rigid_contact_shape1 = rigid_contact_shape1_new, _rigid_contact_point0 = rigid_contact_point0_new, _rigid_contact_point1 = rigid_contact_point1_new, _rigid_contact_offset0 = rigid_contact_offset0_new, _rigid_contact_offset1 = rigid_contact_offset1_new, _rigid_contact_normal = rigid_contact_normal_new, _rigid_contact_thickness = rigid_contact_thickness_new, _rigid_contact_tids = rigid_contact_tids_new)
        state.update_attributes(_body_q = body_q_new, _body_qd = body_qd_new, _body_f = body_f_new)
    print(f"body_q_new inside step: {state._body_q}")
    print(f"body_qd_new inside step: {state._body_qd}")
    print(f"body_f_new inside step: {state._body_f}")
    state.to_pos_quat()
    return model, state
