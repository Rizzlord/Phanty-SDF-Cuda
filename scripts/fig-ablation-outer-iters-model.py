from context import *
import argparse

# Source: https://ten-thousand-models.appspot.com/detail.html?file_id=375275
mesh = "375275"
gs = 100
batch_size_ours = 200000

# outer_iters_list = [0, 1, 10, 20, 50, 100, 200, 500, 1000]
outer_iters_list = [0, 1, 10, 20, 50, 100, 200]

rng_seed = 919993   

# We set up paths and parameters here. Probably a good idea to also put here our method's main parameters so that tuning is easier.
results_path = "results/ablation-outer-iters/"
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


# mnm
# cone = utility.LieConeSDFReconstruction(np.concatenate([U,S[:,None]],axis=1),
#                                     filter_type=3,cut_bbx_factor=1.,filter_results=False,
#                                    psr_screening_weight=1.)
# V_mnm = cone.V
# F_mnm = cone.F
# start_time = time.time()
# V_mnm1, F_mnm1 = utility.kohlbrenner_reconstruction(U, S, method='cones')
# mnm1_time = time.time() - start_time

# start_time = time.time()
# V_mnm2, F_mnm2 = utility.kohlbrenner_reconstruction(U, S, method='RC', V_gt=V_gt, F_gt=F_gt)
# mnm2_time = time.time() - start_time

# # Reach for the Arcs (uncomment if you want to wait a loooong time, or if you reduce the grid size significantly)
# start_time = time.time()
# V_rfta, F_rfta = gpy.reach_for_the_arcs(U, S, verbose=True)
# # pre-loaded rfta
# # V_rfta, F_rfta = gpy.read_mesh( results_path + "rfta.obj")
# # rotate
# V_rfta = V_rfta @ R
# rfta_time = time.time() - start_time

# dual contouring
opts_dc = {"method": contouring._contouring_cpp_module.ContouringMethod.DualContouring}
start_time = time.time()
verts_dc, faces_dc = contouring.py_contouring(
    S, U, gs+1, gs+1, gs+1, 0.0, opts_dc, None, None
)
dc_time = time.time() - start_time


# dual contouring with true grads
def true_sdf(p):
    p = np.array(p)
    # Check that p is just a single point
    assert p.ndim == 1 and p.shape[0] == 3, "true_sdf expects a single 3D point"
    x = p[None, :]   # Add batch dimension so that sdf can process it
    return sdf(x)[0]   # Return the scalar distance (first element of the batch)

sdf_model = contouring._contouring_cpp_module.SDFObject()
sdf_model.f = true_sdf
sdf_model.grad = utility.finite_difference_gradient(sdf_model.f, 1e-6)

opts_dc_true = {"method": contouring._contouring_cpp_module.ContouringMethod.DualContouring}
start_time = time.time()
verts_dc_true, faces_dc_true = contouring.py_contouring(
    S, U, gs+1, gs+1, gs+1, 0.0, opts_dc_true, sdf_model.f, sdf_model.grad
)
dc_true_time = time.time() - start_time


# Write output
# gpy.write_mesh( results_path + f"{mesh}_rfta.obj", V_rfta @ np.linalg.inv(R), F_rfta)
gpy.write_mesh( results_path + f"{mesh}_dc.obj", verts_dc @ np.linalg.inv(R), faces_dc)
gpy.write_mesh( results_path + f"{mesh}_gt.obj", V_gt @ np.linalg.inv(R), F_gt)
# gpy.write_mesh( results_path + f"{mesh}_mnm1.obj", V_mnm1 @ np.linalg.inv(R), F_mnm1)
# gpy.write_mesh( results_path + f"{mesh}_mnm2.obj", V_mnm2 @ np.linalg.inv(R), F_mnm2)
gpy.write_mesh( results_path + f"{mesh}_dc-true-grads.obj", verts_dc_true @ np.linalg.inv(R), faces_dc_true)

# Show it in polyscope along with gt
ps.init()
ps.register_surface_mesh("gt", V_gt @ np.linalg.inv(R), F_gt)
# ps.register_surface_mesh("rfta", V_rfta @ np.linalg.inv(R), F_rfta)
ps.register_surface_mesh("dc", verts_dc @ np.linalg.inv(R), faces_dc)
ps.register_surface_mesh("dc-true-grads", verts_dc_true @ np.linalg.inv(R), faces_dc_true)
# ps.register_surface_mesh("mnm1", V_mnm1 @ np.linalg.inv(R), F_mnm1)
# ps.register_surface_mesh("mnm2", V_mnm2 @ np.linalg.inv(R), F_mnm2)
ps.show()


# ours
for outer_iters in outer_iters_list:
    opts_ours = {"method": contouring._contouring_cpp_module.ContouringMethod.Ours,
            "outer_iters": outer_iters,
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
    gpy.write_mesh( results_path + f"{mesh}_ot{outer_iters}_ours.obj", verts_ours @ np.linalg.inv(R), faces_ours)


