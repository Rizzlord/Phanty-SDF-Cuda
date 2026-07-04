from context import *
import argparse

# Great wall of China: https://www.thingiverse.com/thing:22360/files
mesh = "great_wall_of_china"
gs = 150
batch_size_ours = 200000

rng_seed = 919993   

# We set up paths and parameters here. Probably a good idea to also put here our method's main parameters so that tuning is easier.
results_path = "results/limitations/"
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Load ground truth mesh
filename = "data/" + mesh + ".obj"
V_gt,F_gt = gpy.read_mesh(filename)
V_gt = gpy.normalize_points(V_gt) * 1.5
# sometimes shapes are annoying and they come oriented in different ways. It's best to align them to have the right "up" direction first, so that when we render later we don't need to fiddle with the camera too much. This is what this Rflip does
Rflip = utility.build_rotation_matrix(np.radians(-90), axis="x") @ utility.build_rotation_matrix(np.radians(90), axis="y")
# Rflip = np.eye(3) # if it's already aligned
V_gt = V_gt @ Rflip
# Now the shape is in its canonical orientation, which often is aligned with the world axes. To avoid grid bias, we now rotate V by a random amount along a random axis (we will undo this rotation when saving the results to disk and visualizing them, so that the shapes appear in their canonical orientation)
np.random.seed(rng_seed)
axis = np.random.rand(3)
axis = axis / np.linalg.norm(axis)
angle = np.random.rand() * 2 * np.pi
R = utility.axis_angle_rotation_matrix(axis, angle)
# identity for testing
# R = np.eye(3)
V_gt = V_gt @ R
# Create and abstract SDF function that is the only connection to the shape
sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]
U = contouring.build_grid((gs+1, gs+1, gs+1))
S = sdf(U)


gpy.write_mesh( results_path + f"{mesh}_gt.obj", V_gt @ np.linalg.inv(R), F_gt)

# ours
opts_ours = {"method": contouring._contouring_cpp_module.ContouringMethod.Ours,
        "outer_iters": 100,
        "inner_iters": 100,
        "hermite_update": True,
        "new_hermite_pos_weight": 0.2,
        "new_face_pos_weight": 0.2,
        "new_hermite_normal_weight": 0.2,
        "mu": 0.1,
        "dc_weight": 0.02,
        "verbose": True,
        "batch_size": batch_size_ours
        }
start_time = time.time()
verts_ours, faces_ours = contouring.py_contouring(
    S, U, gs+1, gs+1, gs+1, 0.0, opts_ours, None, None
)
ours_time = time.time() - start_time
gpy.write_mesh( results_path + f"{mesh}_ours.obj", verts_ours @ np.linalg.inv(R), faces_ours)


ps.init()
ps.register_surface_mesh("gt", V_gt @ np.linalg.inv(R), F_gt)
ps.register_surface_mesh("ours", verts_ours @ np.linalg.inv(R), faces_ours)
ps.show()