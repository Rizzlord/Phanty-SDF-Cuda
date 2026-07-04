# This script replicates the results from Figure 4
from context import *

rng_seed = 40

# We set up paths and parameters here. Probably a good idea to also put here our method's main parameters so that tuning is easier.
results_path = "results/dc-fails-if-hermite-bad/"
if not os.path.exists(results_path):
    os.makedirs(results_path)
gs = 50 # grid size
mesh = "amogus"   #abc10
batch_size = 100000
outer_iterations = 20
inner_iterations = 50

# probably best to move this to utility module if you have free time
def axis_angle_rotation_matrix(axis, angle):
    """
    Create a rotation matrix corresponding to the rotation around a general axis by a specified angle.

    R = dd^T + cos(theta)*(I - dd^T) + sin(theta)*skew(d)

    Parameters:
    axis : array
        Axis around which to rotate.
    angle : float
        Angle, in radians, by which to rotate.

    Returns:
    numpy.ndarray
        A rotation matrix.
    """

    # Ensure the axis is a unit vector
    axis = axis / np.linalg.norm(axis)

    # Components of the axis vector
    x, y, z = axis

    # Construct the skew-symmetric matrix
    skew_sym = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

    # Identity matrix
    I = np.eye(3)

    # Outer product of the axis vector with itself
    outer = np.outer(axis, axis)

    # Rotation matrix
    R = outer + np.cos(angle) * (I - outer) + np.sin(angle) * skew_sym

    return R


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
angle = np.radians(20)
R = axis_angle_rotation_matrix(axis, angle)
# identity for testing
# R = np.eye(3)
V_gt = V_gt @ R
U = contouring.build_grid((gs+1, gs+1, gs+1))
# not vectorized, do it one at a time
sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]
S = sdf(U)

# Marching cubes
V_mc, F_mc = gpy.marching_cubes(S, U, gs+1, gs+1, gs+1)

# mnm
# cone = utility.LieConeSDFReconstruction(np.concatenate([U,S[:,None]],axis=1),
#                                     filter_type=3,cut_bbx_factor=1.,filter_results=False,
#                                    psr_screening_weight=1.)
# V_mnm = cone.V
# F_mnm = cone.F
# V_mnm1, F_mnm1 = utility.kohlbrenner_reconstruction(U, S, method='cones')
# V_mnm2, F_mnm2 = utility.kohlbrenner_reconstruction(U, S, method='RC', V_gt=V_gt, F_gt=F_gt)
# V_mnm3, F_mnm3 = utility.kohlbrenner_reconstruction(U, S, method='RC', V_gt=V_gt, F_gt=F_gt, delaunay=True)

# Reach for the Arcs (uncomment if you want to wait a loooong time, or if you reduce the grid size significantly)
# V_rfta, F_rfta = gpy.reach_for_the_arcs(U, S, verbose=True)
# pre-loaded rfta
# V_rfta, F_rfta = gpy.read_mesh( results_path + "rfta.obj")
# rotate
# V_rfta = V_rfta @ R

# dual contouring
opts_dc = {"method": contouring._contouring_cpp_module.ContouringMethod.DualContouring}
verts_dc, faces_dc = contouring.py_contouring(
    S, U, gs+1, gs+1, gs+1, 0.0, opts_dc, None, None
)
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
# start_time = time.time()
verts_dc_true, faces_dc_true = contouring.py_contouring(
    S, U, gs+1, gs+1, gs+1, 0.0, opts_dc_true, sdf_model.f, sdf_model.grad
)
# dc_true_time = time.time() - start_time
# ours
opts_ours = {"method": contouring._contouring_cpp_module.ContouringMethod.Ours,
        "outer_iters": outer_iterations,
        "inner_iters": inner_iterations,
        "hermite_update": True,
        "new_hermite_pos_weight": 1.0,
        "new_face_pos_weight": 1.0,
        "new_hermite_normal_weight": 1.0,
        "mu": 0.4,
        "dc_weight": 0.2,
        "verbose": True,
        "batch_size": batch_size
        }
verts_ours, faces_ours = contouring.py_contouring(
    S, U, gs+1, gs+1, gs+1, 0.0, opts_ours, None, None
)

# mc
# opts_mc = {"method": contouring._contouring_cpp_module.ContouringMethod.MarchingCubes}
# verts_mc, faces_mc = contouring.py_contouring(
#     S, U, gs+1, gs+1, gs+1, 0.0, opts_mc, None, None
# )
# Write output
gpy.write_mesh( results_path + "ours.obj", verts_ours @ np.linalg.inv(R), faces_ours)
gpy.write_mesh( results_path + "gt_dc.obj", verts_dc_true @ np.linalg.inv(R), faces_dc_true)
# gpy.write_mesh( results_path + "rfta.obj", V_rfta @ np.linalg.inv(R), F_rfta)
gpy.write_mesh( results_path + "gt.obj", V_gt @ np.linalg.inv(R), F_gt)
gpy.write_mesh( results_path + "mc.obj", V_mc @ np.linalg.inv(R), F_mc)
gpy.write_mesh( results_path + "dc.obj", verts_dc @ np.linalg.inv(R), faces_dc)
# gpy.write_mesh( results_path + "mnm1.obj", V_mnm1 @ np.linalg.inv(R), F_mnm1)
# gpy.write_mesh( results_path + "mnm2.obj", V_mnm2 @ np.linalg.inv(R), F_mnm2)
# gpy.write_mesh( results_path + "delaunay_mnm2.obj", V_mnm3 @ np.linalg.inv(R), F_mnm3)

ps.init()
# ps.register_surface_mesh("RFTA", V_rfta @ np.linalg.inv(R), F_rfta, enabled=False)
ps.register_surface_mesh("dc", verts_dc @ np.linalg.inv(R), faces_dc, enabled=False)
ps.register_surface_mesh("dc_with_gt", verts_dc_true @ np.linalg.inv(R), faces_dc_true, enabled=False)
ps.register_surface_mesh("ours", verts_ours @ np.linalg.inv(R), faces_ours, enabled=True)
ps.register_surface_mesh("gt", V_gt @ np.linalg.inv(R), F_gt, enabled=False)
ps.register_surface_mesh("mc", V_mc @ np.linalg.inv(R), F_mc, enabled=False)
# ps.register_surface_mesh("mnm1", V_mnm1 @ np.linalg.inv(R), F_mnm1, enabled=False)
# ps.register_surface_mesh("mnm2", V_mnm2 @ np.linalg.inv(R), F_mnm2, enabled=True)
# ps.register_surface_mesh("delaunay_mnm2", V_mnm3 @ np.linalg.inv(R), F_mnm3, enabled=True)
ps.show()
