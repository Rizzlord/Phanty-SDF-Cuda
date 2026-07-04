from context import *

rng_seed = 919993

# Source: https://ten-thousand-models.appspot.com/detail.html?file_id=108613

# We set up paths and parameters here. Probably a good idea to also put here our method's main parameters so that tuning is easier.
results_path = "results/mind-the-gap/"
if not os.path.exists(results_path):
    os.makedirs(results_path)
gs = 100 # grid size
mesh = "108613"
batch_size_ours = 200000

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


start_time = time.time()
V_mnm1, F_mnm1 = utility.kohlbrenner_reconstruction(U, S, method='cones')
mnm1_time = time.time() - start_time

start_time = time.time()
V_mnm2, F_mnm2 = utility.kohlbrenner_reconstruction(U, S, method='RC', V_gt=V_gt, F_gt=F_gt)
mnm2_time = time.time() - start_time

# Marching cubes
start_time = time.time()
V_mc, F_mc = gpy.marching_cubes(S, U, gs+1, gs+1, gs+1)
mc_time = time.time() - start_time

# Reach for the Arcs (uncomment if you want to wait a loooong time, or if you reduce the grid size significantly)
start_time = time.time()
V_rfta, F_rfta = gpy.reach_for_the_arcs(U, S, verbose=True)
rfta_time = time.time() - start_time

# dual contouring
opts_dc = {"method": contouring._contouring_cpp_module.ContouringMethod.DualContouring}
start_time = time.time()
verts_dc, faces_dc = contouring.py_contouring(
    S, U, gs+1, gs+1, gs+1, 0.0, opts_dc, None, None
)
dc_time = time.time() - start_time

# ours
opts_ours = {"method": contouring._contouring_cpp_module.ContouringMethod.Ours,
        "outer_iters": 500,
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

# mc
opts_mc = {"method": contouring._contouring_cpp_module.ContouringMethod.MarchingCubes}
start_time = time.time()
verts_mc, faces_mc = contouring.py_contouring(
    S, U, gs+1, gs+1, gs+1, 0.0, opts_mc, None, None
)
mc_time = time.time() - start_time

gpy.write_mesh( results_path + f"{mesh}_ours.obj", verts_ours @ np.linalg.inv(R), faces_ours)
gpy.write_mesh( results_path + f"{mesh}_rfta.obj", V_rfta @ np.linalg.inv(R), F_rfta)
gpy.write_mesh( results_path + f"{mesh}_dual_contouring.obj", verts_dc @ np.linalg.inv(R), faces_dc)
gpy.write_mesh( results_path + f"{mesh}_ground_truth.obj", V_gt @ np.linalg.inv(R), F_gt)
gpy.write_mesh( results_path + f"{mesh}_marching_cubes.obj", V_mc @ np.linalg.inv(R), F_mc)
gpy.write_mesh( results_path + f"{mesh}_mnm1.obj", V_mnm1 @ np.linalg.inv(R), F_mnm1)
gpy.write_mesh( results_path + f"{mesh}_mnm2.obj", V_mnm2 @ np.linalg.inv(R), F_mnm2)

print(f"Timings for grid size {gs}^3:")
print(f"  - Marching Cubes: {mc_time:.4f} seconds")
print(f"  - Reach For The Arcs: {rfta_time:.4f} seconds")
print(f"  - Dual Contouring: {dc_time:.4f} seconds")
print(f"  - Kohlbrenner Cones: {mnm1_time:.4f} seconds")
print(f"  - Kohlbrenner RC: {mnm2_time:.4f} seconds")
print(f"  - Ours: {ours_time:.4f} seconds")


ps.init()
ps.register_surface_mesh("RFTA", V_rfta @ np.linalg.inv(R), F_rfta)
ps.register_surface_mesh("dc", verts_dc @ np.linalg.inv(R), faces_dc)
ps.register_surface_mesh("ours", verts_ours @ np.linalg.inv(R), faces_ours)
ps.register_surface_mesh("gt", V_gt @ np.linalg.inv(R), F_gt)
ps.register_surface_mesh("mc", verts_mc @ np.linalg.inv(R), faces_mc)
ps.show()