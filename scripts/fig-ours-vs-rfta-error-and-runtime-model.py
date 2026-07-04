from context import *
import argparse

# Model source: https://www.thingiverse.com/thing:5739684

rng_seed = 919993

# We set up paths and parameters here. Probably a good idea to also put here our method's main parameters so that tuning is easier.
results_path = "results/ours-vs-rfta-error-and-runtime/"
if not os.path.exists(results_path):
    os.makedirs(results_path)


p = argparse.ArgumentParser(description="Generate meshes for the dragon figure")
p.add_argument("--gs", type=int, default=50, help="grid size (default: 75)")
p.add_argument("--bs-rfta", dest="batch_size_rfta", type=int, default=10000, help="batch size for Reach For The Arcs (default: 10000)")
p.add_argument("--bs-ours", dest="batch_size_ours", type=int, default=100000, help="batch size for our method (default: 100000)")
args = p.parse_args()

# values (can be provided via CLI)
gs = args.gs
mesh = "Rathalos_Head_Low_Poly"
batch_size_rfta = args.batch_size_rfta
batch_size_ours = args.batch_size_ours

# Load ground truth mesh
filename = "data/" + mesh + ".obj"
V_gt,F_gt = gpy.read_mesh(filename)
V_gt = gpy.normalize_points(V_gt)
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

# Reach for the Arcs (uncomment if you want to wait a loooong time, or if you reduce the grid size significantly)
start_time = time.time()
V_rfta, F_rfta = gpy.reach_for_the_arcs(U, S, verbose=True, batch_size=batch_size_rfta)
rfta_time = time.time() - start_time

# ours
opts_ours = {"method": contouring._contouring_cpp_module.ContouringMethod.Ours,
        "outer_iters": 100,
        "inner_iters": 100,
        "hermite_update": True,
        "new_hermite_pos_weight": 0.2,
        "new_face_pos_weight": 0.2,
        "new_hermite_normal_weight": 0.2,
        "mu": 0.1,
        "dc_weight": 0.01,
        "verbose": True,
        "batch_size": batch_size_ours
        }
start_time = time.time()
verts_ours, faces_ours = contouring.py_contouring(
    S, U, gs+1, gs+1, gs+1, 0.0, opts_ours, None, None
)
ours_time = time.time() - start_time


# Write output
gpy.write_mesh( results_path + f"{mesh}_gs{gs}_bs{batch_size_ours}_ours.obj", verts_ours @ np.linalg.inv(R), faces_ours)
gpy.write_mesh( results_path + f"{mesh}_gs{gs}_bs{batch_size_rfta}_rfta.obj", V_rfta @ np.linalg.inv(R), F_rfta)

# Print timings
print(f"Timings for grid size {gs}^3:")
print(f"  - Reach For The Arcs: {rfta_time:.4f} seconds")
print(f"  - Ours: {ours_time:.4f} seconds")

with open(results_path + f"{mesh}_timings.csv", mode='a', newline='') as timing_file:
    timing_writer = csv.writer(timing_file)
    timing_writer.writerow([gs, "RFTA", batch_size_rfta, f"{rfta_time:.4f}"])
    timing_writer.writerow([gs, "Ours", batch_size_ours, f"{ours_time:.4f}"])

# ps.init()
# ps.register_surface_mesh("RFTA", V_rfta @ np.linalg.inv(R), F_rfta)
# ps.register_surface_mesh("ours", verts_ours @ np.linalg.inv(R), faces_ours)
# ps.register_surface_mesh("gt", V_gt @ np.linalg.inv(R), F_gt)
# ps.show()