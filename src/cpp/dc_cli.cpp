#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <cmath>
#include <algorithm>

#include "dc_backend.h"

#ifdef DC_ENABLE_CUDA
#include "dc_cuda.h"
#include "mesh_to_sdf.h"
#endif

void write_obj(const std::string& filepath, const DualContouringMesh& mesh) {
    std::ofstream out(filepath);
    if (!out) {
        std::cerr << "Failed to open output file: " << filepath << "\n";
        return;
    }
    for (size_t i = 0; i < mesh.vertices.size() / 3; ++i) {
        out << "v " << mesh.vertices[3 * i + 0] << " " << mesh.vertices[3 * i + 1] << " " << mesh.vertices[3 * i + 2] << "\n";
    }
    for (size_t i = 0; i < mesh.faces.size() / 4; ++i) {
        out << "f " << mesh.faces[4 * i + 0] + 1 << " " << mesh.faces[4 * i + 1] + 1 << " " << mesh.faces[4 * i + 2] + 1 << " " << mesh.faces[4 * i + 3] + 1 << "\n";
    }
}

bool read_sdf(const std::string& filepath, DenseSdfGrid& grid) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in) return false;
    in.read(reinterpret_cast<char*>(&grid.nx), sizeof(int));
    in.read(reinterpret_cast<char*>(&grid.ny), sizeof(int));
    in.read(reinterpret_cast<char*>(&grid.nz), sizeof(int));
    in.read(reinterpret_cast<char*>(&grid.vx), sizeof(float));
    in.read(reinterpret_cast<char*>(&grid.vy), sizeof(float));
    in.read(reinterpret_cast<char*>(&grid.vz), sizeof(float));
    in.read(reinterpret_cast<char*>(&grid.ox), sizeof(float));
    in.read(reinterpret_cast<char*>(&grid.oy), sizeof(float));
    in.read(reinterpret_cast<char*>(&grid.oz), sizeof(float));
    int total = grid.nx * grid.ny * grid.nz;
    grid.values.resize(total);
    in.read(reinterpret_cast<char*>(grid.values.data()), total * sizeof(float));
    return in.good();
}

void generate_sphere(DenseSdfGrid& grid, int res) {
    grid.nx = res;
    grid.ny = res;
    grid.nz = res;
    grid.vx = 2.0f / (res - 1);
    grid.vy = 2.0f / (res - 1);
    grid.vz = 2.0f / (res - 1);
    grid.ox = -1.0f;
    grid.oy = -1.0f;
    grid.oz = -1.0f;
    grid.values.resize(res * res * res);
    for (int k = 0; k < res; ++k) {
        float z = grid.oz + k * grid.vz;
        for (int j = 0; j < res; ++j) {
            float y = grid.oy + j * grid.vy;
            for (int i = 0; i < res; ++i) {
                float x = grid.ox + i * grid.vx;
                float dist = std::sqrt(x*x + y*y + z*z);
                grid.values[i + res * (j + res * k)] = dist - 0.5f;
            }
        }
    }
}

void generate_box(DenseSdfGrid& grid, int res) {
    grid.nx = res;
    grid.ny = res;
    grid.nz = res;
    grid.vx = 2.0f / (res - 1);
    grid.vy = 2.0f / (res - 1);
    grid.vz = 2.0f / (res - 1);
    grid.ox = -1.0f;
    grid.oy = -1.0f;
    grid.oz = -1.0f;
    grid.values.resize(res * res * res);
    for (int k = 0; k < res; ++k) {
        float z = grid.oz + k * grid.vz;
        for (int j = 0; j < res; ++j) {
            float y = grid.oy + j * grid.vy;
            for (int i = 0; i < res; ++i) {
                float x = grid.ox + i * grid.vx;
                float dx = std::abs(x) - 0.5f;
                float dy = std::abs(y) - 0.5f;
                float dz = std::abs(z) - 0.5f;
                grid.values[i + res * (j + res * k)] = std::max(dx, std::max(dy, dz));
            }
        }
    }
}

void generate_plane(DenseSdfGrid& grid, int res) {
    grid.nx = res;
    grid.ny = res;
    grid.nz = res;
    grid.vx = 2.0f / (res - 1);
    grid.vy = 2.0f / (res - 1);
    grid.vz = 2.0f / (res - 1);
    grid.ox = -1.0f;
    grid.oy = -1.0f;
    grid.oz = -1.0f;
    grid.values.resize(res * res * res);
    for (int k = 0; k < res; ++k) {
        float z = grid.oz + k * grid.vz;
        for (int j = 0; j < res; ++j) {
            for (int i = 0; i < res; ++i) {
                grid.values[i + res * (j + res * k)] = z;
            }
        }
    }
}

#ifdef DC_ENABLE_CUDA
static bool read_obj_to_sdf(const std::string& filepath, int res, DenseSdfGrid& grid) {
    std::ifstream in(filepath);
    if (!in) return false;

    std::vector<float> vertices;
    std::vector<int> faces;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line.size() >= 2 && line[0] == 'v' && line[1] == ' ') {
            float x, y, z;
            if (sscanf(line.c_str(), "v %f %f %f", &x, &y, &z) == 3) {
                vertices.push_back(x);
                vertices.push_back(y);
                vertices.push_back(z);
            }
        } else if (line.size() >= 2 && line[0] == 'f' && line[1] == ' ') {
            std::vector<int> face_indices;
            char* ptr = (char*)line.c_str() + 2;
            while (*ptr) {
                while (*ptr == ' ' || *ptr == '\t') ptr++;
                if (!*ptr) break;
                int idx = atoi(ptr);
                if (idx != 0) {
                    if (idx < 0) idx = (int)(vertices.size() / 3) + idx + 1;
                    face_indices.push_back(idx - 1);
                }
                while (*ptr && *ptr != ' ' && *ptr != '\t') ptr++;
            }
            if (face_indices.size() >= 3) {
                for (size_t k = 1; k + 1 < face_indices.size(); ++k) {
                    faces.push_back(face_indices[0]);
                    faces.push_back(face_indices[k]);
                    faces.push_back(face_indices[k + 1]);
                }
            }
        }
    }

    if (vertices.empty() || faces.empty()) return false;

    float min_x = vertices[0], max_x = vertices[0];
    float min_y = vertices[1], max_y = vertices[1];
    float min_z = vertices[2], max_z = vertices[2];
    for (size_t i = 0; i < vertices.size() / 3; ++i) {
        float x = vertices[3 * i + 0];
        float y = vertices[3 * i + 1];
        float z = vertices[3 * i + 2];
        if (x < min_x) min_x = x; if (x > max_x) max_x = x;
        if (y < min_y) min_y = y; if (y > max_y) max_y = y;
        if (z < min_z) min_z = z; if (z > max_z) max_z = z;
    }

    float diag = std::sqrt((max_x - min_x)*(max_x - min_x) + (max_y - min_y)*(max_y - min_y) + (max_z - min_z)*(max_z - min_z));
    float pad = (diag > 0.0f) ? (diag * 0.08f) : 0.1f;
    min_x -= pad; max_x += pad;
    min_y -= pad; max_y += pad;
    min_z -= pad; max_z += pad;

    grid.nx = res;
    grid.ny = res;
    grid.nz = res;
    grid.ox = min_x;
    grid.oy = min_y;
    grid.oz = min_z;
    grid.vx = (max_x - min_x) / (res - 1);
    grid.vy = (max_y - min_y) / (res - 1);
    grid.vz = (max_z - min_z) / (res - 1);

    int num_vertices = (int)(vertices.size() / 3);
    int num_faces = (int)(faces.size() / 3);
    DenseSdfGrid computed = compute_mesh_sdf_cuda(
        vertices.data(), num_vertices,
        faces.data(), num_faces,
        res, res, res,
        grid.ox, grid.oy, grid.oz,
        grid.vx, grid.vy, grid.vz
    );
    grid = std::move(computed);
    return true;
}
#endif

int main(int argc, char* argv[]) {
    std::string input_file = "";
    std::string output_file = "";
    std::string backend_type = "cuda-sparse-mvdc";
    std::string gen_type = "";
    std::string normal_mode_str = "finite-difference";
    bool benchmark = false;
    bool close_holes = false;
    bool remove_floaters = false;
    int res = 64;
    int brick_size = 8;
    int chunk_size = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) input_file = argv[++i];
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) output_file = argv[++i];
        } else if (arg == "-b" || arg == "--backend") {
            if (i + 1 < argc) backend_type = argv[++i];
        } else if (arg == "-g" || arg == "--res") {
            if (i + 1 < argc) res = std::stoi(argv[++i]);
        } else if (arg == "-oi" || arg == "-ii" || arg == "-dc" || arg == "-mu" || arg == "-hu") {
            if (i + 1 < argc) i++; // ignore old options safely
        } else if (arg == "--generate") {
            if (i + 1 < argc) gen_type = argv[++i];
        } else if (arg == "--brick-size") {
            if (i + 1 < argc) brick_size = std::stoi(argv[++i]);
        } else if (arg == "--chunk-size") {
            if (i + 1 < argc) chunk_size = std::stoi(argv[++i]);
        } else if (arg == "--normal-mode") {
            if (i + 1 < argc) normal_mode_str = argv[++i];
        } else if (arg == "--benchmark") {
            benchmark = true;
        } else if (arg == "--close-holes") {
            close_holes = true;
        } else if (arg == "--remove-floaters") {
            remove_floaters = true;
        } else {
            if (input_file.empty()) {
                input_file = arg;
            } else if (output_file.empty()) {
                output_file = arg;
            }
        }
    }

    if (output_file.empty() && !input_file.empty()) {
        output_file = input_file;
        input_file = "";
    }

    if (output_file.empty()) {
        std::cerr << "Usage: " << argv[0] << " -i <input.obj|.sdf> -o <output.obj> [options]\n";
        std::cerr << "Options: -b/--backend <cuda-sparse-mvdc|cuda-sparse|cuda|cpu>, -g/--res <int>, --close-holes, --remove-floaters\n";
        return 1;
    }

    DenseSdfGrid grid;
    if (!gen_type.empty()) {
        if (gen_type == "sphere") {
            generate_sphere(grid, res);
        } else if (gen_type == "box") {
            generate_box(grid, res);
        } else if (gen_type == "plane") {
            generate_plane(grid, res);
        } else {
            std::cerr << "Unknown generator type: " << gen_type << "\n";
            return 1;
        }
    } else {
        bool is_obj = (input_file.size() >= 4 && (
            input_file.substr(input_file.size() - 4) == ".obj" ||
            input_file.substr(input_file.size() - 4) == ".OBJ"));

        if (is_obj) {
#ifdef DC_ENABLE_CUDA
            if (!read_obj_to_sdf(input_file, res, grid)) {
                std::cerr << "Failed to convert OBJ to SDF from: " << input_file << "\n";
                return 1;
            }
#else
            std::cerr << "Error: OBJ to SDF conversion requires CUDA enabled.\n";
            return 1;
#endif
        } else {
            if (!read_sdf(input_file, grid)) {
                std::cerr << "Failed to read SDF from: " << input_file << "\n";
                return 1;
            }
        }
    }

    std::unique_ptr<IDualContouringBackend> backend;
#ifdef DC_ENABLE_CUDA
    NormalComputationMode n_mode = NormalComputationMode::FiniteDifference;
    if (normal_mode_str == "edge-gradient") n_mode = NormalComputationMode::EdgeGradient;
    else if (normal_mode_str == "precomputed-gradient") n_mode = NormalComputationMode::PrecomputedGradient;

    if (backend_type == "cuda" || backend_type == "cuda-dense") {
        backend = std::make_unique<CudaDualContouringBackend>();
    } else if (backend_type == "cuda-sparse") {
        backend = std::make_unique<CudaSparseDualContouringBackend>(brick_size, n_mode, chunk_size);
    } else if (backend_type == "cuda-sparse-mvdc" || backend_type == "mvdc") {
        backend = std::make_unique<CudaSparseMvdcDualContouringBackend>(brick_size, n_mode, chunk_size);
    } else {
        backend = std::make_unique<CpuDualContouringBackend>();
    }
#else
    if (backend_type == "cuda" || backend_type == "cuda-dense" || backend_type == "cuda-sparse" || backend_type == "cuda-sparse-mvdc" || backend_type == "mvdc") {
        std::cerr << "Warning: CUDA backend not compiled. Using CPU.\n";
    }
    backend = std::make_unique<CpuDualContouringBackend>();
#endif

    DualContouringStats stats;
    DualContouringMesh mesh = backend->extract(grid, stats);

    if (close_holes || remove_floaters) {
        mesh = postprocess_mesh(mesh, close_holes, remove_floaters);
        stats.vertex_count = (int)(mesh.vertices.size() / 3);
        stats.face_count = (int)(mesh.faces.size() / 4);
    }

    write_obj(output_file, mesh);

    if (benchmark) {
        float active_brick_pct = (stats.total_bricks > 0) ? (100.0f * stats.active_bricks / stats.total_bricks) : 0.0f;
        float active_cell_pct = (stats.total_cells > 0) ? (100.0f * stats.active_cells / stats.total_cells) : 0.0f;
        std::cout << "Backend: " << stats.backend << "\n";
        std::cout << "Grid Dimensions: " << stats.nx << " x " << stats.ny << " x " << stats.nz << "\n";
        std::cout << "Total Cells: " << stats.total_cells << "\n";
        std::cout << "Total Bricks: " << stats.total_bricks << "\n";
        std::cout << "Active Bricks: " << stats.active_bricks << " (" << active_brick_pct << "%)\n";
        std::cout << "Active Cells: " << stats.active_cells << " (" << active_cell_pct << "%)\n";
        std::cout << "Upload Time: " << stats.upload_ms << " ms\n";
        std::cout << "Active Brick Marking Time: " << stats.active_brick_marking_ms << " ms\n";
        std::cout << "Active Brick Compaction Time: " << stats.active_brick_compaction_ms << " ms\n";
        std::cout << "Active Cell Marking Time: " << stats.active_cell_marking_ms << " ms\n";
        std::cout << "Active Cell Compaction Time: " << stats.active_cell_compaction_ms << " ms\n";
        std::cout << "Vertex/QEF Time: " << stats.qef_ms << " ms\n";
        std::cout << "Face Count Time: " << stats.face_count_ms << " ms\n";
        std::cout << "Face Prefix Sum Time: " << stats.face_prefix_sum_ms << " ms\n";
        std::cout << "Face Fill Time: " << stats.face_fill_ms << " ms\n";
        std::cout << "Face Emission Time (Total): " << stats.face_emission_ms << " ms\n";
        std::cout << "Download Time: " << stats.download_ms << " ms\n";
        std::cout << "Total Extraction Time: " << stats.total_ms << " ms\n";
        std::cout << "Vertex Count: " << stats.vertex_count << "\n";
        std::cout << "Face Count: " << stats.face_count << "\n";
        std::cout << "QEF Fallback Count: " << stats.qef_fallback_count << "\n";
        std::cout << "Out-of-Cell Clamps: " << stats.clamp_count << "\n";
        std::cout << "Invalid/NaN Count: " << stats.invalid_count << "\n";
        if (stats.ambiguous_cells > 0 || stats.multi_vertex_cells > 0) {
            std::cout << "Ambiguous Cells: " << stats.ambiguous_cells << "\n";
            std::cout << "Multi-Vertex Cells: " << stats.multi_vertex_cells << "\n";
            std::cout << "One-Vertex Cells: " << stats.one_vertex_cells << "\n";
            std::cout << "Two-Vertex Cells: " << stats.two_vertex_cells << "\n";
            std::cout << "Split Rejection Count: " << stats.split_rejection_count << "\n";
            std::cout << "Bad QEF Count: " << stats.bad_qef_count << "\n";
            std::cout << "Faces Skipped (Missing Cluster): " << stats.faces_skipped_due_to_missing_cluster << "\n";
        }
    }

    return 0;
}
