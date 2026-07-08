#include "mesh_to_sdf.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <chrono>
#include <cmath>
#include <iostream>

#define CUDA_CHECK_SDF(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error in mesh_to_sdf: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
    } \
} while(0)

__device__ __host__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __host__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__ inline float3 operator*(float s, const float3& a) {
    return make_float3(s * a.x, s * a.y, s * a.z);
}

__device__ __host__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ inline float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ float point_to_triangle_distance_sq(
    float3 p, float3 a, float3 b, float3 c,
    float3& out_closest
) {
    float3 ab = b - a;
    float3 ac = c - a;
    float3 ap = p - a;

    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        out_closest = a;
        return dot(ap, ap);
    }

    float3 bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) {
        out_closest = b;
        return dot(bp, bp);
    }

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        out_closest = a + v * ab;
        return dot(p - out_closest, p - out_closest);
    }

    float3 cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) {
        out_closest = c;
        return dot(cp, cp);
    }

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        out_closest = a + w * ac;
        return dot(p - out_closest, p - out_closest);
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        out_closest = b + w * (c - b);
        return dot(p - out_closest, p - out_closest);
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    out_closest = a + v * ab + w * ac;
    return dot(p - out_closest, p - out_closest);
}

__global__ void compute_bin_counts_kernel(
    const float* d_vertices,
    const int* d_faces,
    int num_faces,
    int gx, int gy, int gz,
    float ox, float oy, float oz,
    float bdx, float bdy, float bdz,
    float margin,
    int* d_bin_counts
) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_faces) return;

    int i0 = d_faces[3 * f + 0];
    int i1 = d_faces[3 * f + 1];
    int i2 = d_faces[3 * f + 2];

    float3 v0 = make_float3(d_vertices[3 * i0 + 0], d_vertices[3 * i0 + 1], d_vertices[3 * i0 + 2]);
    float3 v1 = make_float3(d_vertices[3 * i1 + 0], d_vertices[3 * i1 + 1], d_vertices[3 * i1 + 2]);
    float3 v2 = make_float3(d_vertices[3 * i2 + 0], d_vertices[3 * i2 + 1], d_vertices[3 * i2 + 2]);

    float min_x = fminf(v0.x, fminf(v1.x, v2.x)) - margin;
    float min_y = fminf(v0.y, fminf(v1.y, v2.y)) - margin;
    float min_z = fminf(v0.z, fminf(v1.z, v2.z)) - margin;

    float max_x = fmaxf(v0.x, fmaxf(v1.x, v2.x)) + margin;
    float max_y = fmaxf(v0.y, fmaxf(v1.y, v2.y)) + margin;
    float max_z = fmaxf(v0.z, fmaxf(v1.z, v2.z)) + margin;

    int bx0 = max(0, min(gx - 1, (int)floorf((min_x - ox) / bdx)));
    int bx1 = max(0, min(gx - 1, (int)floorf((max_x - ox) / bdx)));
    int by0 = max(0, min(gy - 1, (int)floorf((min_y - oy) / bdy)));
    int by1 = max(0, min(gy - 1, (int)floorf((max_y - oy) / bdy)));
    int bz0 = max(0, min(gz - 1, (int)floorf((min_z - oz) / bdz)));
    int bz1 = max(0, min(gz - 1, (int)floorf((max_z - oz) / bdz)));

    for (int bz = bz0; bz <= bz1; ++bz) {
        for (int by = by0; by <= by1; ++by) {
            for (int bx = bx0; bx <= bx1; ++bx) {
                int bin_idx = bx + gx * (by + gy * bz);
                atomicAdd(&d_bin_counts[bin_idx], 1);
            }
        }
    }
}

__global__ void populate_bins_kernel(
    const float* d_vertices,
    const int* d_faces,
    int num_faces,
    int gx, int gy, int gz,
    float ox, float oy, float oz,
    float bdx, float bdy, float bdz,
    float margin,
    const int* d_bin_offsets,
    int* d_bin_write_counts,
    int* d_bin_triangles
) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_faces) return;

    int i0 = d_faces[3 * f + 0];
    int i1 = d_faces[3 * f + 1];
    int i2 = d_faces[3 * f + 2];

    float3 v0 = make_float3(d_vertices[3 * i0 + 0], d_vertices[3 * i0 + 1], d_vertices[3 * i0 + 2]);
    float3 v1 = make_float3(d_vertices[3 * i1 + 0], d_vertices[3 * i1 + 1], d_vertices[3 * i1 + 2]);
    float3 v2 = make_float3(d_vertices[3 * i2 + 0], d_vertices[3 * i2 + 1], d_vertices[3 * i2 + 2]);

    float min_x = fminf(v0.x, fminf(v1.x, v2.x)) - margin;
    float min_y = fminf(v0.y, fminf(v1.y, v2.y)) - margin;
    float min_z = fminf(v0.z, fminf(v1.z, v2.z)) - margin;

    float max_x = fmaxf(v0.x, fmaxf(v1.x, v2.x)) + margin;
    float max_y = fmaxf(v0.y, fmaxf(v1.y, v2.y)) + margin;
    float max_z = fmaxf(v0.z, fmaxf(v1.z, v2.z)) + margin;

    int bx0 = max(0, min(gx - 1, (int)floorf((min_x - ox) / bdx)));
    int bx1 = max(0, min(gx - 1, (int)floorf((max_x - ox) / bdx)));
    int by0 = max(0, min(gy - 1, (int)floorf((min_y - oy) / bdy)));
    int by1 = max(0, min(gy - 1, (int)floorf((max_y - oy) / bdy)));
    int bz0 = max(0, min(gz - 1, (int)floorf((min_z - oz) / bdz)));
    int bz1 = max(0, min(gz - 1, (int)floorf((max_z - oz) / bdz)));

    for (int bz = bz0; bz <= bz1; ++bz) {
        for (int by = by0; by <= by1; ++by) {
            for (int bx = bx0; bx <= bx1; ++bx) {
                int bin_idx = bx + gx * (by + gy * bz);
                int offset = atomicAdd(&d_bin_write_counts[bin_idx], 1);
                d_bin_triangles[d_bin_offsets[bin_idx] + offset] = f;
            }
        }
    }
}

__global__ void mark_active_bins_kernel(
    int gx, int gy, int gz,
    const int* d_bin_offsets,
    uint8_t* d_bin_active
) {
    int bin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_bins = gx * gy * gz;
    if (bin_idx >= num_bins) return;

    int bx = bin_idx % gx;
    int by = (bin_idx / gx) % gy;
    int bz = bin_idx / (gx * gy);

    bool active = false;
    for (int dz = -1; dz <= 1 && !active; ++dz) {
        for (int dy = -1; dy <= 1 && !active; ++dy) {
            for (int dx = -1; dx <= 1 && !active; ++dx) {
                int nbx = bx + dx;
                int nby = by + dy;
                int nbz = bz + dz;
                if (nbx >= 0 && nbx < gx && nby >= 0 && nby < gy && nbz >= 0 && nbz < gz) {
                    int n_idx = nbx + gx * (nby + gy * nbz);
                    if (d_bin_offsets[n_idx + 1] > d_bin_offsets[n_idx]) {
                        active = true;
                    }
                }
            }
        }
    }
    d_bin_active[bin_idx] = active ? 1 : 0;
}

__global__ void evaluate_sdf_grid_kernel(
    int nx, int ny, int nz,
    float ox, float oy, float oz,
    float vx, float vy, float vz,
    int gx, int gy, int gz,
    float bdx, float bdy, float bdz,
    const int* d_bin_offsets,
    const int* d_bin_triangles,
    const float* d_vertices,
    const int* d_faces,
    const uint8_t* d_bin_active,
    float* d_values
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_voxels = nx * ny * nz;
    if (idx >= total_voxels) return;

    int ix = idx % nx;
    int iy = (idx / nx) % ny;
    int iz = idx / (nx * ny);

    float3 p = make_float3(ox + ix * vx, oy + iy * vy, oz + iz * vz);

    int center_bx = max(0, min(gx - 1, (int)floorf((p.x - ox) / bdx)));
    int center_by = max(0, min(gy - 1, (int)floorf((p.y - oy) / bdy)));
    int center_bz = max(0, min(gz - 1, (int)floorf((p.z - oz) / bdz)));

    int bin_idx_center = center_bx + gx * (center_by + gy * center_bz);
    if (d_bin_active != nullptr && d_bin_active[bin_idx_center] == 0) {
        d_values[idx] = 10.0f;
        return;
    }

    float min_dist_sq = 3.402823466e+38f;
    float3 best_closest = make_float3(0.0f, 0.0f, 0.0f);
    float3 best_normal = make_float3(0.0f, 1.0f, 0.0f);
    int radius = 0;
    int max_radius = 1;
    float min_box_dim = fminf(bdx, fminf(bdy, bdz));

    while (radius <= max_radius) {
        for (int bz = max(0, center_bz - radius); bz <= min(gz - 1, center_bz + radius); ++bz) {
            for (int by = max(0, center_by - radius); by <= min(gy - 1, center_by + radius); ++by) {
                for (int bx = max(0, center_bx - radius); bx <= min(gx - 1, center_bx + radius); ++bx) {
                    if (abs(bx - center_bx) < radius && abs(by - center_by) < radius && abs(bz - center_bz) < radius) {
                        continue;
                    }
                    int bin_idx = bx + gx * (by + gy * bz);
                    int start = d_bin_offsets[bin_idx];
                    int end = d_bin_offsets[bin_idx + 1];

                    for (int i = start; i < end; ++i) {
                        int f = d_bin_triangles[i];
                        int i0 = d_faces[3 * f + 0];
                        int i1 = d_faces[3 * f + 1];
                        int i2 = d_faces[3 * f + 2];

                        float3 v0 = make_float3(d_vertices[3 * i0 + 0], d_vertices[3 * i0 + 1], d_vertices[3 * i0 + 2]);
                        float3 v1 = make_float3(d_vertices[3 * i1 + 0], d_vertices[3 * i1 + 1], d_vertices[3 * i1 + 2]);
                        float3 v2 = make_float3(d_vertices[3 * i2 + 0], d_vertices[3 * i2 + 1], d_vertices[3 * i2 + 2]);

                        float3 closest;
                        float d_sq = point_to_triangle_distance_sq(p, v0, v1, v2, closest);
                        float3 n = cross(v1 - v0, v2 - v0);
                        float n_len = sqrtf(dot(n, n));
                        if (n_len > 1e-12f) {
                            n = make_float3(n.x / n_len, n.y / n_len, n.z / n_len);
                        } else {
                            continue;
                        }

                        if (d_sq < min_dist_sq - 1e-6f * (min_dist_sq + 1e-8f)) {
                            min_dist_sq = d_sq;
                            best_closest = closest;
                            best_normal = n;
                        } else if (fabsf(d_sq - min_dist_sq) <= 1e-6f * (min_dist_sq + 1e-8f)) {
                            best_normal = make_float3(best_normal.x + n.x, best_normal.y + n.y, best_normal.z + n.z);
                        }
                    }
                }
            }
        }
        if (min_dist_sq < (radius * min_box_dim) * (radius * min_box_dim) && radius > 0) {
            break;
        }
        radius++;
    }

    if (min_dist_sq == 3.402823466e+38f) {
        d_values[idx] = 10.0f;
    } else {
        float dist = sqrtf(min_dist_sq);
        float norm_len = sqrtf(dot(best_normal, best_normal));
        if (norm_len > 1e-12f) {
            best_normal = make_float3(best_normal.x / norm_len, best_normal.y / norm_len, best_normal.z / norm_len);
        }
        float best_dot = dot(p - best_closest, best_normal);
        if (best_dot < 0.0f) {
            dist = -dist;
        }
        if (dist > 3.0f * min_box_dim) {
            dist = 10.0f;
        } else if (dist < -3.0f * min_box_dim) {
            dist = -10.0f;
        }
        d_values[idx] = dist;
    }
}

DenseSdfGridDevice compute_mesh_sdf_device_cuda(
    const float* vertices, int num_vertices,
    const int* faces, int num_faces,
    int nx, int ny, int nz,
    float ox, float oy, float oz,
    float vx, float vy, float vz,
    float* out_ms
) {
    auto t0 = std::chrono::high_resolution_clock::now();

    float* d_vertices = nullptr;
    int* d_faces = nullptr;
    CUDA_CHECK_SDF(cudaMalloc(&d_vertices, num_vertices * 3 * sizeof(float)));
    CUDA_CHECK_SDF(cudaMalloc(&d_faces, num_faces * 3 * sizeof(int)));
    CUDA_CHECK_SDF(cudaMemcpy(d_vertices, vertices, num_vertices * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_SDF(cudaMemcpy(d_faces, faces, num_faces * 3 * sizeof(int), cudaMemcpyHostToDevice));

    int gx = max(1, nx / 8);
    int gy = max(1, ny / 8);
    int gz = max(1, nz / 8);
    float bdx = (nx * vx) / gx;
    float bdy = (ny * vy) / gy;
    float bdz = (nz * vz) / gz;
    float margin = 2.0f * fmaxf(vx, fmaxf(vy, vz));

    int num_bins = gx * gy * gz;
    int* d_bin_counts = nullptr;
    int* d_bin_offsets = nullptr;
    int* d_bin_write_counts = nullptr;
    int* d_bin_triangles = nullptr;

    CUDA_CHECK_SDF(cudaMalloc(&d_bin_counts, (num_bins + 1) * sizeof(int)));
    CUDA_CHECK_SDF(cudaMalloc(&d_bin_offsets, (num_bins + 1) * sizeof(int)));
    CUDA_CHECK_SDF(cudaMalloc(&d_bin_write_counts, num_bins * sizeof(int)));
    CUDA_CHECK_SDF(cudaMemset(d_bin_counts, 0, (num_bins + 1) * sizeof(int)));
    CUDA_CHECK_SDF(cudaMemset(d_bin_write_counts, 0, num_bins * sizeof(int)));

    int threads = 256;
    int blocks_f = (num_faces + threads - 1) / threads;
    compute_bin_counts_kernel<<<blocks_f, threads>>>(d_vertices, d_faces, num_faces, gx, gy, gz, ox, oy, oz, bdx, bdy, bdz, margin, d_bin_counts);
    CUDA_CHECK_SDF(cudaGetLastError());

    thrust::device_ptr<int> dev_counts(d_bin_counts);
    thrust::device_ptr<int> dev_offsets(d_bin_offsets);
    thrust::exclusive_scan(dev_counts, dev_counts + num_bins + 1, dev_offsets);

    int total_bin_triangles = 0;
    CUDA_CHECK_SDF(cudaMemcpy(&total_bin_triangles, d_bin_offsets + num_bins, sizeof(int), cudaMemcpyDeviceToHost));

    if (total_bin_triangles > 0) {
        CUDA_CHECK_SDF(cudaMalloc(&d_bin_triangles, total_bin_triangles * sizeof(int)));
        populate_bins_kernel<<<blocks_f, threads>>>(d_vertices, d_faces, num_faces, gx, gy, gz, ox, oy, oz, bdx, bdy, bdz, margin, d_bin_offsets, d_bin_write_counts, d_bin_triangles);
        CUDA_CHECK_SDF(cudaGetLastError());
    }

    uint8_t* d_bin_active = nullptr;
    CUDA_CHECK_SDF(cudaMalloc(&d_bin_active, num_bins * sizeof(uint8_t)));
    int blocks_bins = (num_bins + threads - 1) / threads;
    mark_active_bins_kernel<<<blocks_bins, threads>>>(gx, gy, gz, d_bin_offsets, d_bin_active);
    CUDA_CHECK_SDF(cudaGetLastError());

    float* d_values = nullptr;
    int total_voxels = nx * ny * nz;
    CUDA_CHECK_SDF(cudaMalloc(&d_values, total_voxels * sizeof(float)));

    int blocks_v = (total_voxels + threads - 1) / threads;
    evaluate_sdf_grid_kernel<<<blocks_v, threads>>>(nx, ny, nz, ox, oy, oz, vx, vy, vz, gx, gy, gz, bdx, bdy, bdz, d_bin_offsets, d_bin_triangles, d_vertices, d_faces, d_bin_active, d_values);
    CUDA_CHECK_SDF(cudaGetLastError());
    CUDA_CHECK_SDF(cudaDeviceSynchronize());

    CUDA_CHECK_SDF(cudaFree(d_bin_active));
    if (d_bin_triangles) CUDA_CHECK_SDF(cudaFree(d_bin_triangles));
    CUDA_CHECK_SDF(cudaFree(d_bin_write_counts));
    CUDA_CHECK_SDF(cudaFree(d_bin_offsets));
    CUDA_CHECK_SDF(cudaFree(d_bin_counts));
    CUDA_CHECK_SDF(cudaFree(d_faces));
    CUDA_CHECK_SDF(cudaFree(d_vertices));

    auto t1 = std::chrono::high_resolution_clock::now();
    if (out_ms) {
        *out_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    }

    DenseSdfGridDevice dev_grid;
    dev_grid.nx = nx;
    dev_grid.ny = ny;
    dev_grid.nz = nz;
    dev_grid.ox = ox;
    dev_grid.oy = oy;
    dev_grid.oz = oz;
    dev_grid.vx = vx;
    dev_grid.vy = vy;
    dev_grid.vz = vz;
    dev_grid.d_values = d_values;
    return dev_grid;
}

DenseSdfGrid compute_mesh_sdf_cuda(
    const float* vertices, int num_vertices,
    const int* faces, int num_faces,
    int nx, int ny, int nz,
    float ox, float oy, float oz,
    float vx, float vy, float vz,
    float* out_ms
) {
    DenseSdfGridDevice dev_grid = compute_mesh_sdf_device_cuda(vertices, num_vertices, faces, num_faces, nx, ny, nz, ox, oy, oz, vx, vy, vz, out_ms);

    DenseSdfGrid host_grid;
    host_grid.nx = nx;
    host_grid.ny = ny;
    host_grid.nz = nz;
    host_grid.ox = ox;
    host_grid.oy = oy;
    host_grid.oz = oz;
    host_grid.vx = vx;
    host_grid.vy = vy;
    host_grid.vz = vz;
    host_grid.values.resize(nx * ny * nz);

    CUDA_CHECK_SDF(cudaMemcpy(host_grid.values.data(), dev_grid.d_values, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost));
    free_mesh_sdf_device_cuda(dev_grid);

    return host_grid;
}

void free_mesh_sdf_device_cuda(DenseSdfGridDevice& device_grid) {
    if (device_grid.d_values) {
        cudaFree((void*)device_grid.d_values);
        device_grid.d_values = nullptr;
    }
}
