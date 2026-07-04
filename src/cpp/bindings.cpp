#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>

#include "contouring.h"
#include "sdf.h"
#include <igl/grid.h>

namespace py = pybind11;


// Create a grid of positions in [min_corner, max_corner] with given resolutions
static Eigen::MatrixXd py_grid(
	const Eigen::RowVector3d& min_corner,
	const Eigen::RowVector3d& max_corner,
	int nx,
	int ny,
	int nz) 
{
	Eigen::MatrixXd GV;
	igl::grid(Eigen::RowVector3i(nx, ny, nz), GV);

	Eigen::RowVector3d extents = max_corner - min_corner;
	for (int i = 0; i < GV.rows(); ++i) {
		GV.row(i).array() *= extents.array();
		GV.row(i) += min_corner;
	}
	return GV;
}


// Wrapper for contouring overload without true sdf/grad
static py::tuple py_contouring(
	const Eigen::VectorXd& S,
	const Eigen::MatrixXd& GV,
	int resX,
	int resY,
	int resZ,
	double isoValue,
	const ContouringOptions& options
) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	contouring(S, GV, resX, resY, resZ, isoValue, V, F, options);
	return py::make_tuple(V, F);
}

// Wrapper for contouring overload with true sdf and grad
static py::tuple py_contouring_with_sdf(
	const Eigen::VectorXd& S,
	const Eigen::MatrixXd& GV,
	int resX,
	int resY,
	int resZ,
	double isoValue,
	const ContouringOptions& options,
	const SDF& true_sdf,
	const SDFGrad& true_sdf_grad
) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	contouring(S, GV, resX, resY, resZ, isoValue, V, F, options, true_sdf, true_sdf_grad);
	return py::make_tuple(V, F);
}

PYBIND11_MODULE(_contouring_cpp_module, m) {
	m.doc() = "Python bindings for pseudosdf-contouring-cpp";

	py::enum_<ContouringMethod>(m, "ContouringMethod")
		.value("MarchingCubes", ContouringMethod::MarchingCubes)
		.value("DualContouring", ContouringMethod::DualContouring)
		.value("Ours", ContouringMethod::Ours)
		.export_values();

	py::class_<ContouringOptions>(m, "ContouringOptions")
		.def(py::init<>())
		.def_readwrite("method", &ContouringOptions::method)
		.def_readwrite("verbose", &ContouringOptions::verbose)
		.def_readwrite("mu", &ContouringOptions::mu)
		.def_readwrite("dc_weight", &ContouringOptions::dc_weight)
		.def_readwrite("sphere_weight", &ContouringOptions::sphere_weight)
		.def_readwrite("svd_threshold", &ContouringOptions::svd_threshold)
		.def_readwrite("outer_iters", &ContouringOptions::outer_iters)
		.def_readwrite("inner_iters", &ContouringOptions::inner_iters)
		.def_readwrite("hermite_update", &ContouringOptions::hermite_update)
		.def_readwrite("batch_size", &ContouringOptions::batch_size)
		.def_readwrite("new_hermite_pos_weight", &ContouringOptions::new_hermite_pos_weight)
		.def_readwrite("new_face_pos_weight", &ContouringOptions::new_face_pos_weight)
		.def_readwrite("new_hermite_normal_weight", &ContouringOptions::new_hermite_normal_weight);

	py::class_<SDFObject>(m, "SDFObject")
		.def(py::init<>())
		.def_readwrite("f", &SDFObject::f)
		.def_readwrite("grad", &SDFObject::grad);

    m.def("_make_rotated_box_sdf",
	    (SDFObject(*)(const Eigen::RowVector3d&, const Eigen::Matrix3d&, double)) &make_rotated_box_sdf,
		  "Create a rotated box SDF at the origin",
		  py::arg("half_extents"), py::arg("R"), py::arg("fd_h") = -1.0);

    m.def("_make_rotated_box_sdf_center",
	    (SDFObject(*)(const Eigen::RowVector3d&, const Eigen::RowVector3d&, const Eigen::Matrix3d&, double)) &make_rotated_box_sdf,
		  "Create a rotated box SDF specifying the center",
		  py::arg("half_extents"), py::arg("center"), py::arg("R"), py::arg("fd_h") = -1.0);

	m.def("_make_torus_sdf", &make_torus_sdf, py::arg("ra"), py::arg("rb"), py::arg("fd_h"));
	m.def("_make_cylinder_sdf", &make_cylinder_sdf, py::arg("he"), py::arg("r"), py::arg("fd_h"));
	m.def("_make_triangular_prism_sdf", &make_triangular_prism_sdf, py::arg("half_extents"), py::arg("R"), py::arg("fd_h"));
	m.def("_make_cut_sphere_sdf", &make_cut_sphere_sdf, py::arg("r"), py::arg("h"), py::arg("fd_h"));
	m.def("_make_octahedron_sdf", &make_octahedron_sdf, py::arg("s"), py::arg("R"), py::arg("fd_h"));

	m.def("_grid", &py_grid, "Generate a grid of positions", py::arg("min_corner"), py::arg("max_corner"), py::arg("nx"), py::arg("ny"), py::arg("nz"));

    m.def("_contouring_cpp", &py_contouring, "Internal: run contouring (no true sdf)",
	    py::arg("S"), py::arg("GV"), py::arg("resX"), py::arg("resY"), py::arg("resZ"), py::arg("isoValue"), py::arg("options"));

    m.def("_contouring_cpp_with_sdf", &py_contouring_with_sdf, "Internal: run contouring with true sdf and grad",
	    py::arg("S"), py::arg("GV"), py::arg("resX"), py::arg("resY"), py::arg("resZ"), py::arg("isoValue"), py::arg("options"), py::arg("true_sdf"), py::arg("true_sdf_grad"));
}