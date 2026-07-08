#pragma once

#include <vector>
#include <string>
#include <memory>

class ISdfSampler {
public:
    virtual ~ISdfSampler() = default;
    virtual float sample(float x, float y, float z) const = 0;
};

class DenseSdfGrid : public ISdfSampler {
public:
    int nx, ny, nz;
    float vx, vy, vz;
    float ox, oy, oz;
    std::vector<float> values;

    float sample(float x, float y, float z) const override;
};

struct DualContouringMesh {
    std::vector<float> vertices;
    std::vector<int> faces;
};

class ISdfDeviceSampler {
public:
    virtual ~ISdfDeviceSampler() = default;
};

class DenseGridSampler : public ISdfDeviceSampler {
public:
    int nx, ny, nz;
    float vx, vy, vz;
    float ox, oy, oz;
    const float* d_values;
};

struct DualContouringStats {
    std::string backend;
    int nx, ny, nz;
    int total_cells;
    int active_cells;
    int total_bricks;
    int active_bricks;
    float upload_ms;
    float marking_ms;
    float compaction_ms;
    float active_brick_marking_ms;
    float active_brick_compaction_ms;
    float active_cell_marking_ms;
    float active_cell_compaction_ms;
    float qef_ms;
    float face_emission_ms;
    float face_count_ms;
    float face_prefix_sum_ms;
    float face_fill_ms;
    float download_ms;
    float total_ms;
    int vertex_count;
    int face_count;
    int qef_fallback_count;
    int clamp_count;
    int invalid_count;
    DualContouringStats() : backend(""), nx(0), ny(0), nz(0), total_cells(0), active_cells(0), total_bricks(0), active_bricks(0), upload_ms(0.0f), marking_ms(0.0f), compaction_ms(0.0f), active_brick_marking_ms(0.0f), active_brick_compaction_ms(0.0f), active_cell_marking_ms(0.0f), active_cell_compaction_ms(0.0f), qef_ms(0.0f), face_emission_ms(0.0f), face_count_ms(0.0f), face_prefix_sum_ms(0.0f), face_fill_ms(0.0f), download_ms(0.0f), total_ms(0.0f), vertex_count(0), face_count(0), qef_fallback_count(0), clamp_count(0), invalid_count(0) {}
};

class IDualContouringBackend {
public:
    virtual ~IDualContouringBackend() = default;
    virtual DualContouringMesh extract(const DenseSdfGrid& grid, DualContouringStats& stats) = 0;
};

class CpuDualContouringBackend : public IDualContouringBackend {
public:
    DualContouringMesh extract(const DenseSdfGrid& grid, DualContouringStats& stats) override;
};
