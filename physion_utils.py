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
