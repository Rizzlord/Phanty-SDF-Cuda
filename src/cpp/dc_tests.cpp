#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <memory>
#include <algorithm>

#include "dc_backend.h"

#ifdef DC_ENABLE_CUDA
#include "dc_cuda.h"
#endif

void run_sphere_test(IDualContouringBackend* cpu, IDualContouringBackend* gpu) {
    DenseSdfGrid grid;
    int res = 32;
    grid.nx = res; grid.ny = res; grid.nz = res;
    grid.vx = 2.0f / (res - 1); grid.vy = 2.0f / (res - 1); grid.vz = 2.0f / (res - 1);
    grid.ox = -1.0f; grid.oy = -1.0f; grid.oz = -1.0f;
    grid.values.resize(res * res * res);
    for (int k = 0; k < res; ++k) {
        float z = grid.oz + k * grid.vz;
        for (int j = 0; j < res; ++j) {
            float y = grid.oy + j * grid.vy;
            for (int i = 0; i < res; ++i) {
                float x = grid.ox + i * grid.vx;
                grid.values[i + res * (j + res * k)] = std::sqrt(x*x + y*y + z*z) - 0.5f;
            }
        }
    }

    DualContouringStats cpu_stats;
    DualContouringMesh cpu_mesh = cpu->extract(grid, cpu_stats);

    std::cout << "[Sphere CPU] Vertices: " << cpu_stats.vertex_count << ", Faces: " << cpu_stats.face_count << "\n";

    if (gpu) {
        DualContouringStats gpu_stats;
        DualContouringMesh gpu_mesh = gpu->extract(grid, gpu_stats);
        std::cout << "[Sphere GPU] Vertices: " << gpu_stats.vertex_count << ", Faces: " << gpu_stats.face_count << "\n";
        assert(gpu_stats.vertex_count == cpu_stats.active_cells);
        assert(gpu_stats.face_count == cpu_mesh.faces.size() / 4);
        for (float v : gpu_mesh.vertices) {
            assert(!std::isnan(v) && !std::isinf(v));
        }
        for (int f : gpu_mesh.faces) {
            assert(f >= 0 && f < gpu_stats.vertex_count);
        }
    }
}

void run_box_test(IDualContouringBackend* cpu, IDualContouringBackend* gpu) {
    DenseSdfGrid grid;
    int res = 32;
    grid.nx = res; grid.ny = res; grid.nz = res;
    grid.vx = 2.0f / (res - 1); grid.vy = 2.0f / (res - 1); grid.vz = 2.0f / (res - 1);
    grid.ox = -1.0f; grid.oy = -1.0f; grid.oz = -1.0f;
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

    DualContouringStats cpu_stats;
    DualContouringMesh cpu_mesh = cpu->extract(grid, cpu_stats);

    std::cout << "[Box CPU] Vertices: " << cpu_stats.vertex_count << ", Faces: " << cpu_stats.face_count << "\n";

    if (gpu) {
        DualContouringStats gpu_stats;
        DualContouringMesh gpu_mesh = gpu->extract(grid, gpu_stats);
        std::cout << "[Box GPU] Vertices: " << gpu_stats.vertex_count << ", Faces: " << gpu_stats.face_count << "\n";
        assert(gpu_stats.vertex_count == cpu_stats.active_cells);
        assert(gpu_stats.face_count == cpu_mesh.faces.size() / 4);
        for (float v : gpu_mesh.vertices) {
            assert(!std::isnan(v) && !std::isinf(v));
        }
        for (int f : gpu_mesh.faces) {
            assert(f >= 0 && f < gpu_stats.vertex_count);
        }
    }
}

void run_plane_test(IDualContouringBackend* cpu, IDualContouringBackend* gpu) {
    DenseSdfGrid grid;
    int res = 16;
    grid.nx = res; grid.ny = res; grid.nz = res;
    grid.vx = 2.0f / (res - 1); grid.vy = 2.0f / (res - 1); grid.vz = 2.0f / (res - 1);
    grid.ox = -1.0f; grid.oy = -1.0f; grid.oz = -1.0f;
    grid.values.resize(res * res * res);
    for (int k = 0; k < res; ++k) {
        float z = grid.oz + k * grid.vz;
        for (int j = 0; j < res; ++j) {
            for (int i = 0; i < res; ++i) {
                grid.values[i + res * (j + res * k)] = z;
            }
        }
    }

    DualContouringStats cpu_stats;
    DualContouringMesh cpu_mesh = cpu->extract(grid, cpu_stats);

    std::cout << "[Plane CPU] Vertices: " << cpu_stats.vertex_count << ", Faces: " << cpu_stats.face_count << "\n";

    if (gpu) {
        DualContouringStats gpu_stats;
        DualContouringMesh gpu_mesh = gpu->extract(grid, gpu_stats);
        std::cout << "[Plane GPU] Vertices: " << gpu_stats.vertex_count << ", Faces: " << gpu_stats.face_count << "\n";
        std::cout << "[Plane GPU] Fallbacks: " << gpu_stats.qef_fallback_count << "\n";
        assert(gpu_stats.vertex_count == cpu_stats.active_cells);
        assert(gpu_stats.face_count == cpu_mesh.faces.size() / 4);
        assert(gpu_stats.qef_fallback_count > 0);
        for (float v : gpu_mesh.vertices) {
            assert(!std::isnan(v) && !std::isinf(v));
        }
        for (int f : gpu_mesh.faces) {
            assert(f >= 0 && f < gpu_stats.vertex_count);
        }
    }
}

void run_thin_gap_test(IDualContouringBackend* cpu, IDualContouringBackend* gpu) {
    DenseSdfGrid grid;
    int res = 32;
    grid.nx = res; grid.ny = res; grid.nz = res;
    grid.vx = 2.0f / (res - 1); grid.vy = 2.0f / (res - 1); grid.vz = 2.0f / (res - 1);
    grid.ox = -1.0f; grid.oy = -1.0f; grid.oz = -1.0f;
    grid.values.resize(res * res * res);
    for (int k = 0; k < res; ++k) {
        float z = grid.oz + k * grid.vz;
        for (int j = 0; j < res; ++j) {
            float y = grid.oy + j * grid.vy;
            for (int i = 0; i < res; ++i) {
                float x = grid.ox + i * grid.vx;
                float d1 = std::sqrt((x-0.2f)*(x-0.2f) + y*y + z*z) - 0.25f;
                float d2 = std::sqrt((x+0.2f)*(x+0.2f) + y*y + z*z) - 0.25f;
                grid.values[i + res * (j + res * k)] = std::min(d1, d2);
            }
        }
    }

    DualContouringStats cpu_stats;
    DualContouringMesh cpu_mesh = cpu->extract(grid, cpu_stats);

    std::cout << "[Thin Gap CPU] Vertices: " << cpu_stats.vertex_count << ", Faces: " << cpu_stats.face_count << "\n";

    if (gpu) {
        DualContouringStats gpu_stats;
        DualContouringMesh gpu_mesh = gpu->extract(grid, gpu_stats);
        std::cout << "[Thin Gap GPU] Vertices: " << gpu_stats.vertex_count << ", Faces: " << gpu_stats.face_count << "\n";
        assert(gpu_stats.vertex_count == cpu_stats.active_cells);
        assert(gpu_stats.face_count == cpu_mesh.faces.size() / 4);
        for (float v : gpu_mesh.vertices) {
            assert(!std::isnan(v) && !std::isinf(v));
        }
        for (int f : gpu_mesh.faces) {
            assert(f >= 0 && f < gpu_stats.vertex_count);
        }
    }
}

void run_empty_grid_test(IDualContouringBackend* cpu, IDualContouringBackend* gpu) {
    DenseSdfGrid grid;
    int res = 16;
    grid.nx = res; grid.ny = res; grid.nz = res;
    grid.vx = 2.0f / (res - 1); grid.vy = 2.0f / (res - 1); grid.vz = 2.0f / (res - 1);
    grid.ox = -1.0f; grid.oy = -1.0f; grid.oz = -1.0f;
    grid.values.assign(res * res * res, 1.0f);

    DualContouringStats cpu_stats;
    DualContouringMesh cpu_mesh = cpu->extract(grid, cpu_stats);

    assert(cpu_stats.vertex_count == 0);
    assert(cpu_stats.face_count == 0);

    if (gpu) {
        DualContouringStats gpu_stats;
        DualContouringMesh gpu_mesh = gpu->extract(grid, gpu_stats);
        assert(gpu_stats.vertex_count == 0);
        assert(gpu_stats.face_count == 0);
    }
    std::cout << "[Empty Grid] Passed\n";
}

void run_full_neg_grid_test(IDualContouringBackend* cpu, IDualContouringBackend* gpu) {
    DenseSdfGrid grid;
    int res = 16;
    grid.nx = res; grid.ny = res; grid.nz = res;
    grid.vx = 2.0f / (res - 1); grid.vy = 2.0f / (res - 1); grid.vz = 2.0f / (res - 1);
    grid.ox = -1.0f; grid.oy = -1.0f; grid.oz = -1.0f;
    grid.values.assign(res * res * res, -1.0f);

    DualContouringStats cpu_stats;
    DualContouringMesh cpu_mesh = cpu->extract(grid, cpu_stats);

    assert(cpu_stats.vertex_count == 0);
    assert(cpu_stats.face_count == 0);

    if (gpu) {
        DualContouringStats gpu_stats;
        DualContouringMesh gpu_mesh = gpu->extract(grid, gpu_stats);
        assert(gpu_stats.vertex_count == 0);
        assert(gpu_stats.face_count == 0);
    }
    std::cout << "[Full Negative Grid] Passed\n";
}

int main() {
    std::unique_ptr<IDualContouringBackend> cpu = std::make_unique<CpuDualContouringBackend>();
    std::unique_ptr<IDualContouringBackend> gpu;

#ifdef DC_ENABLE_CUDA
    gpu = std::make_unique<CudaDualContouringBackend>();
    std::cout << "CUDA backend enabled in tests\n";
#else
    std::cout << "CUDA backend disabled in tests\n";
#endif

    run_sphere_test(cpu.get(), gpu.get());
    run_box_test(cpu.get(), gpu.get());
    run_plane_test(cpu.get(), gpu.get());
    run_thin_gap_test(cpu.get(), gpu.get());
    run_empty_grid_test(cpu.get(), gpu.get());
    run_full_neg_grid_test(cpu.get(), gpu.get());

    std::cout << "ALL TESTS PASSED\n";
    return 0;
}
