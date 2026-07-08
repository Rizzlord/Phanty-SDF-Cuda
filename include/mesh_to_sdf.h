#pragma once

#include "dc_backend.h"
#include "dc_cuda.h"

DenseSdfGrid compute_mesh_sdf_cuda(
    const float* vertices, int num_vertices,
    const int* faces, int num_faces,
    int nx, int ny, int nz,
    float ox, float oy, float oz,
    float vx, float vy, float vz,
    float* out_ms = nullptr,
    bool voxelize_first = false,
    int voxel_res = 256
);

DenseSdfGridDevice compute_mesh_sdf_device_cuda(
    const float* vertices, int num_vertices,
    const int* faces, int num_faces,
    int nx, int ny, int nz,
    float ox, float oy, float oz,
    float vx, float vy, float vz,
    float* out_ms = nullptr,
    bool voxelize_first = false,
    int voxel_res = 256
);

void free_mesh_sdf_device_cuda(DenseSdfGridDevice& device_grid);
