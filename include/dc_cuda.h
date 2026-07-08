#pragma once

#include "dc_backend.h"

enum class NormalComputationMode {
    FiniteDifference = 0,
    EdgeGradient = 1,
    PrecomputedGradient = 2
};

struct BrickCoord {
    int bx, by, bz;
};

struct BrickInfo {
    int bx, by, bz;
    int start_ix, start_iy, start_iz;
};

struct ActiveBrick {
    int brick_idx;
};

struct DenseSdfGridDevice {
    int nx, ny, nz;
    float vx, vy, vz;
    float ox, oy, oz;
    const float* d_values;
};

class CudaDualContouringBackend : public IDualContouringBackend {
public:
    DualContouringMesh extract(const DenseSdfGrid& grid, DualContouringStats& stats) override;
    DualContouringMesh extract_device(const DenseSdfGridDevice& grid, DualContouringStats& stats);
};

class CudaSparseDualContouringBackend : public IDualContouringBackend {
public:
    int brick_size;
    NormalComputationMode normal_mode;
    int chunk_size;
    bool multi_vertex_cells;

    CudaSparseDualContouringBackend(
        int b_size = 8,
        NormalComputationMode n_mode = NormalComputationMode::FiniteDifference,
        int c_size = 0,
        bool mv_cells = false
    ) : brick_size(b_size), normal_mode(n_mode), chunk_size(c_size), multi_vertex_cells(mv_cells) {}

    DualContouringMesh extract(const DenseSdfGrid& grid, DualContouringStats& stats) override;
    DualContouringMesh extract_device(const DenseSdfGridDevice& grid, DualContouringStats& stats);
};

class CudaSparseMvdcDualContouringBackend : public CudaSparseDualContouringBackend {
public:
    CudaSparseMvdcDualContouringBackend(
        int b_size = 8,
        NormalComputationMode n_mode = NormalComputationMode::FiniteDifference,
        int c_size = 0
    ) : CudaSparseDualContouringBackend(b_size, n_mode, c_size, true) {}
};
