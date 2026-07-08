#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include "contouring.h"
#include "sdf.h"
#include <igl/grid.h>
#include "dc_backend.h"
#ifdef DC_ENABLE_CUDA
#include "dc_cuda.h"
#include "mesh_to_sdf.h"
#endif

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

    py::class_<DenseSdfGrid>(m, "DenseSdfGrid")
        .def(py::init<>())
        .def_readwrite("nx", &DenseSdfGrid::nx)
        .def_readwrite("ny", &DenseSdfGrid::ny)
        .def_readwrite("nz", &DenseSdfGrid::nz)
        .def_readwrite("vx", &DenseSdfGrid::vx)
        .def_readwrite("vy", &DenseSdfGrid::vy)
        .def_readwrite("vz", &DenseSdfGrid::vz)
        .def_readwrite("ox", &DenseSdfGrid::ox)
        .def_readwrite("oy", &DenseSdfGrid::oy)
        .def_readwrite("oz", &DenseSdfGrid::oz)
        .def_readwrite("values", &DenseSdfGrid::values);

    py::class_<DualContouringMesh>(m, "DualContouringMesh")
        .def(py::init<>())
        .def_readwrite("vertices", &DualContouringMesh::vertices)
        .def_readwrite("faces", &DualContouringMesh::faces);

    py::class_<DualContouringStats>(m, "DualContouringStats")
        .def(py::init<>())
        .def_readwrite("backend", &DualContouringStats::backend)
        .def_readwrite("nx", &DualContouringStats::nx)
        .def_readwrite("ny", &DualContouringStats::ny)
        .def_readwrite("nz", &DualContouringStats::nz)
        .def_readwrite("total_cells", &DualContouringStats::total_cells)
        .def_readwrite("active_cells", &DualContouringStats::active_cells)
        .def_readwrite("total_bricks", &DualContouringStats::total_bricks)
        .def_readwrite("active_bricks", &DualContouringStats::active_bricks)
        .def_readwrite("upload_ms", &DualContouringStats::upload_ms)
        .def_readwrite("marking_ms", &DualContouringStats::marking_ms)
        .def_readwrite("compaction_ms", &DualContouringStats::compaction_ms)
        .def_readwrite("active_brick_marking_ms", &DualContouringStats::active_brick_marking_ms)
        .def_readwrite("active_brick_compaction_ms", &DualContouringStats::active_brick_compaction_ms)
        .def_readwrite("active_cell_marking_ms", &DualContouringStats::active_cell_marking_ms)
        .def_readwrite("active_cell_compaction_ms", &DualContouringStats::active_cell_compaction_ms)
        .def_readwrite("qef_ms", &DualContouringStats::qef_ms)
        .def_readwrite("face_emission_ms", &DualContouringStats::face_emission_ms)
        .def_readwrite("face_count_ms", &DualContouringStats::face_count_ms)
        .def_readwrite("face_prefix_sum_ms", &DualContouringStats::face_prefix_sum_ms)
        .def_readwrite("face_fill_ms", &DualContouringStats::face_fill_ms)
        .def_readwrite("download_ms", &DualContouringStats::download_ms)
        .def_readwrite("total_ms", &DualContouringStats::total_ms)
        .def_readwrite("vertex_count", &DualContouringStats::vertex_count)
        .def_readwrite("face_count", &DualContouringStats::face_count)
        .def_readwrite("qef_fallback_count", &DualContouringStats::qef_fallback_count)
        .def_readwrite("clamp_count", &DualContouringStats::clamp_count)
        .def_readwrite("invalid_count", &DualContouringStats::invalid_count)
        .def_readwrite("ambiguous_cells", &DualContouringStats::ambiguous_cells)
        .def_readwrite("multi_vertex_cells", &DualContouringStats::multi_vertex_cells)
        .def_readwrite("one_vertex_cells", &DualContouringStats::one_vertex_cells)
        .def_readwrite("two_vertex_cells", &DualContouringStats::two_vertex_cells)
        .def_readwrite("split_rejection_count", &DualContouringStats::split_rejection_count)
        .def_readwrite("bad_qef_count", &DualContouringStats::bad_qef_count)
        .def_readwrite("faces_skipped_due_to_missing_cluster", &DualContouringStats::faces_skipped_due_to_missing_cluster);

    py::class_<CpuDualContouringBackend>(m, "CpuDualContouringBackend")
        .def(py::init<>())
        .def("extract", [](CpuDualContouringBackend& self, const DenseSdfGrid& grid) {
            DualContouringStats stats;
            DualContouringMesh mesh = self.extract(grid, stats);
            return py::make_tuple(mesh, stats);
        });

#ifdef DC_ENABLE_CUDA
    py::class_<DenseSdfGridDevice>(m, "DenseSdfGridDevice")
        .def(py::init<>())
        .def_readwrite("nx", &DenseSdfGridDevice::nx)
        .def_readwrite("ny", &DenseSdfGridDevice::ny)
        .def_readwrite("nz", &DenseSdfGridDevice::nz)
        .def_readwrite("vx", &DenseSdfGridDevice::vx)
        .def_readwrite("vy", &DenseSdfGridDevice::vy)
        .def_readwrite("vz", &DenseSdfGridDevice::vz)
        .def_readwrite("ox", &DenseSdfGridDevice::ox)
        .def_readwrite("oy", &DenseSdfGridDevice::oy)
        .def_readwrite("oz", &DenseSdfGridDevice::oz)
        .def("free", [](DenseSdfGridDevice& self) {
            free_mesh_sdf_device_cuda(self);
        });

    py::enum_<NormalComputationMode>(m, "NormalComputationMode")
        .value("FiniteDifference", NormalComputationMode::FiniteDifference)
        .value("EdgeGradient", NormalComputationMode::EdgeGradient)
        .value("PrecomputedGradient", NormalComputationMode::PrecomputedGradient)
        .export_values();

    py::class_<CudaDualContouringBackend>(m, "CudaDualContouringBackend")
        .def(py::init<>())
        .def("extract", [](CudaDualContouringBackend& self, const DenseSdfGrid& grid) {
            DualContouringStats stats;
            DualContouringMesh mesh = self.extract(grid, stats);
            return py::make_tuple(mesh, stats);
        })
        .def("extract_device", [](CudaDualContouringBackend& self, const DenseSdfGridDevice& grid) {
            DualContouringStats stats;
            DualContouringMesh mesh = self.extract_device(grid, stats);
            return py::make_tuple(mesh, stats);
        });

    py::class_<CudaSparseDualContouringBackend>(m, "CudaSparseDualContouringBackend")
        .def(py::init<int, NormalComputationMode, int, bool>(),
            py::arg("brick_size") = 8,
            py::arg("normal_mode") = NormalComputationMode::FiniteDifference,
            py::arg("chunk_size") = 0,
            py::arg("multi_vertex_cells") = false)
        .def_readwrite("brick_size", &CudaSparseDualContouringBackend::brick_size)
        .def_readwrite("normal_mode", &CudaSparseDualContouringBackend::normal_mode)
        .def_readwrite("chunk_size", &CudaSparseDualContouringBackend::chunk_size)
        .def_readwrite("multi_vertex_cells", &CudaSparseDualContouringBackend::multi_vertex_cells)
        .def("extract", [](CudaSparseDualContouringBackend& self, const DenseSdfGrid& grid) {
            DualContouringStats stats;
            DualContouringMesh mesh = self.extract(grid, stats);
            return py::make_tuple(mesh, stats);
        })
        .def("extract_device", [](CudaSparseDualContouringBackend& self, const DenseSdfGridDevice& grid) {
            DualContouringStats stats;
            DualContouringMesh mesh = self.extract_device(grid, stats);
            return py::make_tuple(mesh, stats);
        });

    py::class_<CudaSparseMvdcDualContouringBackend>(m, "CudaSparseMvdcDualContouringBackend")
        .def(py::init<int, NormalComputationMode, int>(),
            py::arg("brick_size") = 8,
            py::arg("normal_mode") = NormalComputationMode::FiniteDifference,
            py::arg("chunk_size") = 0)
        .def_readwrite("brick_size", &CudaSparseMvdcDualContouringBackend::brick_size)
        .def_readwrite("normal_mode", &CudaSparseMvdcDualContouringBackend::normal_mode)
        .def_readwrite("chunk_size", &CudaSparseMvdcDualContouringBackend::chunk_size)
        .def_readwrite("multi_vertex_cells", &CudaSparseMvdcDualContouringBackend::multi_vertex_cells)
        .def("extract", [](CudaSparseMvdcDualContouringBackend& self, const DenseSdfGrid& grid) {
            DualContouringStats stats;
            DualContouringMesh mesh = self.extract(grid, stats);
            return py::make_tuple(mesh, stats);
        })
        .def("extract_device", [](CudaSparseMvdcDualContouringBackend& self, const DenseSdfGridDevice& grid) {
            DualContouringStats stats;
            DualContouringMesh mesh = self.extract_device(grid, stats);
            return py::make_tuple(mesh, stats);
        });

    m.def("mesh_to_sdf_cuda", [](py::array_t<float> vertices, py::array_t<int> faces, int nx, int ny, int nz, float ox, float oy, float oz, float vx, float vy, float vz) {
        py::buffer_info v_info = vertices.request();
        py::buffer_info f_info = faces.request();
        float ms = 0.0f;
        DenseSdfGrid grid = compute_mesh_sdf_cuda((const float*)v_info.ptr, v_info.shape[0], (const int*)f_info.ptr, f_info.shape[0], nx, ny, nz, ox, oy, oz, vx, vy, vz, &ms);
        return py::make_tuple(grid, ms);
    });

    m.def("mesh_to_sdf_device_cuda", [](py::array_t<float> vertices, py::array_t<int> faces, int nx, int ny, int nz, float ox, float oy, float oz, float vx, float vy, float vz) {
        py::buffer_info v_info = vertices.request();
        py::buffer_info f_info = faces.request();
        float ms = 0.0f;
        DenseSdfGridDevice dev_grid = compute_mesh_sdf_device_cuda((const float*)v_info.ptr, v_info.shape[0], (const int*)f_info.ptr, f_info.shape[0], nx, ny, nz, ox, oy, oz, vx, vy, vz, &ms);
        return py::make_tuple(dev_grid, ms);
    });
#endif
}