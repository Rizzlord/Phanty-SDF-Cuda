from context import *

# This is a direct adaptation of the C++ cell_diagonal function
def cell_diagonal(GV, resX, resY, resZ):
    min_x = np.min(GV[:, 0])
    min_y = np.min(GV[:, 1])
    min_z = np.min(GV[:, 2])

    dx = (np.max(GV[:, 0]) - min_x) / (resX - 1)
    dy = (np.max(GV[:, 1]) - min_y) / (resY - 1)
    dz = (np.max(GV[:, 2]) - min_z) / (resZ - 1)

    h = min(dx, dy, dz)
    dim_sqrt = np.sqrt(3.0)
    cell_diag = h * dim_sqrt

    return cell_diag


rng_seed = 40

results_path = "results/narrow-band/"
if not os.path.exists(results_path):
    os.makedirs(results_path)
gs = 100 # grid size
# https://www.thingiverse.com/thing:759413/files
#mesh = "buster"
# Stanford bunny
mesh = "bunny"
batch_size_ours = 200000
n = 10


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

# cell_d = cell_diagonal(U, gs+1, gs+1, gs+1)
# print(f"Cell diagonal: {cell_d}")

# # Set all the data in S with |SDF| > n * cell_d to inf (or -inf, depending on the sign)
# print(f"Setting values with |SDF| > {n} * cell_d = {n * cell_d} to inf")
# print(f"Number of affected voxels: {np.sum(np.abs(S) > n * cell_d)} out of {S.size} total voxels")
# S[np.abs(S) > n * cell_d] = np.sign(S[np.abs(S) > n * cell_d]) * np.inf

# not_infty = U[np.abs(S) <= n * cell_d]

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


# Write output
gpy.write_mesh( results_path + f"{mesh}_all_data_ours.obj", verts_ours @ np.linalg.inv(R), faces_ours)
gpy.write_mesh( results_path + f"{mesh}_n{n}_ours.obj", verts_ours @ np.linalg.inv(R), faces_ours)
gpy.write_mesh( results_path + f"{mesh}_gt.obj", V_gt @ np.linalg.inv(R), F_gt)

V_gt, _, _, F_gt, _, _ = igl.readOBJ(str(ROOT / "results" / "narrow-band" / f"{mesh}_gt.obj"))
verts_all_data_ours, _, _, faces_all_data_ours, _, _ = igl.readOBJ(str(ROOT / "results" / "narrow-band" / f"{mesh}_all_data_ours.obj"))
verts_n_ours, _, _, faces_n_ours, _, _ = igl.readOBJ(str(ROOT / "results" / "narrow-band" / f"{mesh}_n{n}_ours.obj"))


ps.init()
#ps.register_point_cloud("points used", not_infty, color=(0.3, 0.3, 0.8))
ps.register_surface_mesh("all data", verts_all_data_ours, faces_all_data_ours)
ps.register_surface_mesh(f"n={n}", verts_n_ours, faces_n_ours)
ps.register_surface_mesh("gt", V_gt, F_gt)
ps.show()