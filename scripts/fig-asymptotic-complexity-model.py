from context import *

rng_seed = 919993

# Source: https://ten-thousand-models.appspot.com/detail.html?file_id=69079

# We set up paths and parameters here. Probably a good idea to also put here our method's main parameters so that tuning is easier.
results_path = "results/asymptotic-complexity/"
if not os.path.exists(results_path):
    os.makedirs(results_path)

mesh = "69079"
batch_size_ours = 200000
timings_csv_path = os.path.join(results_path, f"{mesh}_timings.csv")

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

grid_sizes = [10, 25, 50, 75, 100, 125, 150, 200]
baseline_max_grid_size = 125

# Create and abstract SDF function that is the only connection to the shape
sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]

gpy.write_mesh( results_path + f"{mesh}_gt.obj", V_gt @ np.linalg.inv(R), F_gt)

timing_rows = []

for gs in grid_sizes:
    run_baselines = gs <= baseline_max_grid_size
    U = contouring.build_grid((gs+1, gs+1, gs+1))
    S = sdf(U)

    if run_baselines:
        start_time = time.time()
        V_mnm1, F_mnm1 = utility.kohlbrenner_reconstruction(U, S, method='cones')
        mnm1_time = time.time() - start_time

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
    # opts_ours = {"method": contouring._contouring_cpp_module.ContouringMethod.Ours,
    #         "outer_iters": 100,
    #         "inner_iters": 100,
    #         "hermite_update": True,
    #         "new_hermite_pos_weight": 0.2,
    #         "new_face_pos_weight": 0.2,
    #         "new_hermite_normal_weight": 0.2,
    #         "mu": 0.1,
    #         "dc_weight": 0.02,
    #         "verbose": True,
    #         "batch_size": batch_size_ours
    #         }
    # start_time = time.time()
    # verts_ours, faces_ours = contouring.py_contouring(
    #     S, U, gs+1, gs+1, gs+1, 0.0, opts_ours, None, None
    # )
    # ours_time = time.time() - start_time

    # Print timings
    print(f"Timings for grid size {gs}^3:")
    if run_baselines:
        print(f"  - Reach For The Arcs: {rfta_time:.4f} seconds")
        print(f"  - Kohlbrenner Cones: {mnm1_time:.4f} seconds")
    else:
        print(f"  - Baselines skipped for grid sizes > {baseline_max_grid_size}")
    print(f"  - Dual Contouring: {dc_time:.4f} seconds")
    #print(f"  - Ours: {ours_time:.4f} seconds")

    if run_baselines:
        timing_rows.extend([
            {
                "grid_size": gs,
                "num_grid_cells": gs**3,
                "method": "rfta",
                "time_seconds": rfta_time,
            },
            {
                "grid_size": gs,
                "num_grid_cells": gs**3,
                "method": "mnm1",
                "time_seconds": mnm1_time,
            },
        ])
    timing_rows.append({
        "grid_size": gs,
        "num_grid_cells": gs**3,
        "method": "dc",
        "time_seconds": dc_time,
    })
    # timing_rows.append({
    #     "grid_size": gs,
    #     "num_grid_cells": gs**3,
    #     "method": "ours",
    #     "time_seconds": ours_time,
    # })
    pd.DataFrame(timing_rows).to_csv(timings_csv_path, index=False)
    print(f"Saved timings to {timings_csv_path}")

    # Write output
    if run_baselines:
        gpy.write_mesh( results_path + f"{mesh}_gs{gs}_rfta.obj", V_rfta @ np.linalg.inv(R), F_rfta)
        gpy.write_mesh( results_path + f"{mesh}_gs{gs}_mnm1.obj", V_mnm1 @ np.linalg.inv(R), F_mnm1)
    gpy.write_mesh( results_path + f"{mesh}_gs{gs}_dc.obj", verts_dc @ np.linalg.inv(R), faces_dc)
    # gpy.write_mesh( results_path + f"{mesh}_gs{gs}_ours.obj", verts_ours @ np.linalg.inv(R), faces_ours)

    # Show it in polyscope along with gt
    # ps.init()
    # ps.register_surface_mesh("gt", V_gt @ np.linalg.inv(R), F_gt)
    # ps.register_surface_mesh("rfta", V_rfta @ np.linalg.inv(R), F_rfta)
    # ps.register_surface_mesh("dc", verts_dc @ np.linalg.inv(R), faces_dc)
    # ps.register_surface_mesh("mnm1", V_mnm1 @ np.linalg.inv(R), F_mnm1)
    # ps.register_surface_mesh("ours", verts_ours @ np.linalg.inv(R), faces_ours)
    # ps.show()
