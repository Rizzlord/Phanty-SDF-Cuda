from context import *
import time

# Configuration
SEED = 100
GS = 10  
RUN_RFTA = True  # set True if you want to run Reach-for-the-Arcs (can be slow)
RESULTS_DIR = "results/box/"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

np.random.seed(SEED)
rng = np.random.default_rng(SEED)

# Random box parameters
half_extents = list(0.1 + 0.3 * rng.random(3))  # each in [0.1, 0.4]
center = list(-0.2 + 0.4 * rng.random(3))      # each in [-0.2, 0.2]
axis = rng.random(3)
axis = axis / np.linalg.norm(axis)
angle = rng.random() * (np.pi / 6.0)  # up to 30 degrees
R = utility.axis_angle_rotation_matrix(axis, angle)

print("Random box parameters:")
print(f"  half_extents = {half_extents}")
print(f"  center = {center}")
print(f"  axis = {axis}, angle (deg) = {np.degrees(angle):.2f}")

# Build SDF and sample on a grid
box_sdf = utility.make_rotated_box_sdf_center(half_extents, center, R)
U = contouring.build_grid((GS + 1, GS + 1, GS + 1))
print(f"Built grid with {len(U)} sample points (resolution {GS}^3)")

# Evaluate SDF values on the grid
start = time.time()
S = np.array([box_sdf.f(p) for p in U])
time_sdf = time.time() - start
print(f"Evaluated SDF on grid: {time_sdf:.3f}s")

# Marching Cubes (libigl) - direct call to gpy.marching_cubes
opts_mc = {"method": contouring._contouring_cpp_module.ContouringMethod.MarchingCubes}
start = time.time()
try:
    V_mc, F_mc = gpy.marching_cubes(S, U, GS + 1, GS + 1, GS + 1)
    mc_time = time.time() - start
    print(f"Marching Cubes done: {mc_time:.3f}s; V={len(V_mc)}, F={len(F_mc)}")
    gpy.write_mesh(os.path.join(RESULTS_DIR, "box_marching_cubes.obj"), V_mc, F_mc)
except Exception as e:
    mc_time = None
    print(f"Marching Cubes failed: {e}")

# Dual Contouring (binding)
opts_dc = {"method": contouring._contouring_cpp_module.ContouringMethod.DualContouring}
start = time.time()
try:
    verts_dc, faces_dc = contouring.py_contouring(S, U, GS + 1, GS + 1, GS + 1, 0.0, opts_dc, None, None)
    dc_time = time.time() - start
    print(f"Dual Contouring done: {dc_time:.3f}s; V={len(verts_dc)}, F={len(faces_dc)}")
    gpy.write_mesh(os.path.join(RESULTS_DIR, "box_dual_contouring.obj"), verts_dc, faces_dc)
except Exception as e:
    dc_time = None
    print(f"Dual Contouring failed: {e}")

# Ours (iterative) - use reasonable defaults
opts_ours = {
    "method": contouring._contouring_cpp_module.ContouringMethod.Ours,
    "outer_iters": 100,
    "inner_iters": 100,
    "hermite_update": True,
    "new_hermite_pos_weight": 0.2,
    "new_face_pos_weight": 0.2,
    "new_hermite_normal_weight": 0.2,
    "mu": 0.1,
    "dc_weight": 0.02,
    "verbose": False,
}
start = time.time()
try:
    verts_ours, faces_ours = contouring.py_contouring(S, U, GS + 1, GS + 1, GS + 1, 0.0, opts_ours, None, None)
    ours_time = time.time() - start
    print(f"Ours done: {ours_time:.3f}s; V={len(verts_ours)}, F={len(faces_ours)}")
    gpy.write_mesh(os.path.join(RESULTS_DIR, "box_ours.obj"), verts_ours, faces_ours)
except Exception as e:
    ours_time = None
    print(f"Ours method failed: {e}")

# Reach For The Arcs (RFTA) - optional because it can be slow
V_rfta = np.zeros((0, 3))
F_rfta = np.zeros((0, 3), dtype=int)
if RUN_RFTA:
    start = time.time()
    try:
        V_rfta, F_rfta = gpy.reach_for_the_arcs(U, S, verbose=True)
        t_rfta = time.time() - start
        print(f"RFTA done: {t_rfta:.3f}s; V={len(V_rfta)}, F={len(F_rfta)}")
        # triangulate quads returned by RFTA
        try:
            F_rfta = utility.triangulate_quads(V_rfta, F_rfta)
        except Exception:
            pass
        gpy.write_mesh(os.path.join(RESULTS_DIR, "box_rfta.obj"), V_rfta, F_rfta)
    except Exception as e:
        print(f"RFTA failed: {e}")
else:
    print("RFTA skipped (set RUN_RFTA = True at the top to enable)")

# Register and show in Polyscope for quick visualization
ps.init()
if len(V_rfta) > 0:
    ps.register_surface_mesh("RFTA", V_rfta, F_rfta, enabled=False)
if dc_time is not None:
    ps.register_surface_mesh("DualContouring", verts_dc, faces_dc, enabled=False)
if ours_time is not None:
    ps.register_surface_mesh("Ours", verts_ours, faces_ours, enabled=True)
if mc_time is not None:
    ps.register_surface_mesh("MarchingCubes", V_mc, F_mc, enabled=False)

ps.show()

# Write the objs
if mc_time is not None:
    gpy.write_mesh(os.path.join(RESULTS_DIR, "box_mc.obj"), V_mc, F_mc)
if dc_time is not None:
    gpy.write_mesh(os.path.join(RESULTS_DIR, "box_dc.obj"), verts_dc, faces_dc)
if ours_time is not None:
    gpy.write_mesh(os.path.join(RESULTS_DIR, "box_ours.obj"), verts_ours, faces_ours)
if RUN_RFTA and len(V_rfta) > 0:
    gpy.write_mesh(os.path.join(RESULTS_DIR, "box_rfta.obj"), V_rfta, F_rfta)

print(f"Results written to: {RESULTS_DIR}")
