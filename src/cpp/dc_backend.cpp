#include "dc_backend.h"
#include "contouring.h"
#include "Cell.h"
#include <chrono>
#include <algorithm>
#include <cmath>

float DenseSdfGrid::sample(float x, float y, float z) const {
    float lx = (x - ox) / vx;
    float ly = (y - oy) / vy;
    float lz = (z - oz) / vz;

    int x0 = std::max(0, std::min((int)lx, nx - 2));
    int y0 = std::max(0, std::min((int)ly, ny - 2));
    int z0 = std::max(0, std::min((int)lz, nz - 2));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    float tx = lx - x0;
    float ty = ly - y0;
    float tz = lz - z0;
    tx = std::max(0.0f, std::min(1.0f, tx));
    ty = std::max(0.0f, std::min(1.0f, ty));
    tz = std::max(0.0f, std::min(1.0f, tz));

    auto idx = [&](int i, int j, int k) {
        return i + nx * (j + ny * k);
    };

    float c000 = values[idx(x0, y0, z0)];
    float c100 = values[idx(x1, y0, z0)];
    float c010 = values[idx(x0, y1, z0)];
    float c110 = values[idx(x1, y1, z0)];
    float c001 = values[idx(x0, y0, z1)];
    float c101 = values[idx(x1, y0, z1)];
    float c011 = values[idx(x0, y1, z1)];
    float c111 = values[idx(x1, y1, z1)];

    float c00 = c000 * (1.0f - tx) + c100 * tx;
    float c01 = c001 * (1.0f - tx) + c101 * tx;
    float c10 = c010 * (1.0f - tx) + c110 * tx;
    float c11 = c011 * (1.0f - tx) + c111 * tx;

    float c0 = c00 * (1.0f - ty) + c10 * ty;
    float c1 = c01 * (1.0f - ty) + c11 * ty;

    return c0 * (1.0f - tz) + c1 * tz;
}

DualContouringMesh CpuDualContouringBackend::extract(const DenseSdfGrid& grid, DualContouringStats& stats) {
    auto start_time = std::chrono::high_resolution_clock::now();

    int total_vertices = grid.nx * grid.ny * grid.nz;
    Eigen::VectorXd S(total_vertices);
    Eigen::MatrixXd GV(total_vertices, 3);

    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                int flat_idx = i + grid.nx * (j + grid.ny * k);
                S(flat_idx) = grid.values[flat_idx];
                GV(flat_idx, 0) = grid.ox + i * grid.vx;
                GV(flat_idx, 1) = grid.oy + j * grid.vy;
                GV(flat_idx, 2) = grid.oz + k * grid.vz;
            }
        }
    }

    ContouringOptions options;
    options.method = ContouringMethod::DualContouring;
    options.verbose = false;

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    contouring(S, GV, grid.nx, grid.ny, grid.nz, 0.0, V, F, options);

    DualContouringMesh mesh;
    mesh.vertices.resize(V.rows() * 3);
    for (int i = 0; i < V.rows(); ++i) {
        mesh.vertices[3 * i + 0] = (float)V(i, 0);
        mesh.vertices[3 * i + 1] = (float)V(i, 1);
        mesh.vertices[3 * i + 2] = (float)V(i, 2);
    }

    mesh.faces.resize(F.rows() * F.cols());
    for (int i = 0; i < F.rows(); ++i) {
        for (int j = 0; j < F.cols(); ++j) {
            mesh.faces[F.cols() * i + j] = F(i, j);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    float total_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();

    stats.backend = "cpu";
    stats.nx = grid.nx;
    stats.ny = grid.ny;
    stats.nz = grid.nz;
    stats.total_cells = (grid.nx - 1) * (grid.ny - 1) * (grid.nz - 1);
    stats.active_cells = 0;
    stats.upload_ms = 0.0f;
    stats.marking_ms = 0.0f;
    stats.compaction_ms = 0.0f;
    stats.qef_ms = 0.0f;
    stats.face_emission_ms = 0.0f;
    stats.download_ms = 0.0f;
    stats.total_ms = total_ms;
    stats.vertex_count = (int)V.rows();
    stats.face_count = (int)F.rows();
    stats.qef_fallback_count = 0;
    stats.clamp_count = 0;

    std::vector<Cell>* cellsPtr = getGeneratedCells();
    if (cellsPtr) {
        for (const auto& c : *cellsPtr) {
            if (c.has_vertex) {
                stats.active_cells++;
            }
        }
    }

    return mesh;
}
