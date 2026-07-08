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
        d_values[idx] = 3.402823466e+38f;
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
                        float area_2 = sqrtf(dot(n, n));
                        if (area_2 > 1e-12f) {
                            n = make_float3(n.x / area_2, n.y / area_2, n.z / area_2);
                        } else {
                            continue;
                        }

                        float w = area_2;
                        float d_v0_sq = dot(closest - v0, closest - v0);
                        float d_v1_sq = dot(closest - v1, closest - v1);
                        float d_v2_sq = dot(closest - v2, closest - v2);
                        float v_eps_sq = 1e-6f * min_box_dim * min_box_dim;
                        if (d_v0_sq < v_eps_sq) {
                            float3 e1 = v1 - v0, e2 = v2 - v0;
                            float l1 = sqrtf(dot(e1, e1)), l2 = sqrtf(dot(e2, e2));
                            if (l1 > 1e-6f && l2 > 1e-6f) {
                                float c_th = fmaxf(-1.0f, fminf(1.0f, dot(e1, e2) / (l1 * l2)));
                                w = acosf(c_th);
                            }
                        } else if (d_v1_sq < v_eps_sq) {
                            float3 e1 = v0 - v1, e2 = v2 - v1;
                            float l1 = sqrtf(dot(e1, e1)), l2 = sqrtf(dot(e2, e2));
                            if (l1 > 1e-6f && l2 > 1e-6f) {
                                float c_th = fmaxf(-1.0f, fminf(1.0f, dot(e1, e2) / (l1 * l2)));
                                w = acosf(c_th);
                            }
                        } else if (d_v2_sq < v_eps_sq) {
                            float3 e1 = v0 - v2, e2 = v1 - v2;
                            float l1 = sqrtf(dot(e1, e1)), l2 = sqrtf(dot(e2, e2));
                            if (l1 > 1e-6f && l2 > 1e-6f) {
                                float c_th = fmaxf(-1.0f, fminf(1.0f, dot(e1, e2) / (l1 * l2)));
                                w = acosf(c_th);
                            }
                        }

                        if (d_sq < min_dist_sq - 1e-5f * (min_dist_sq + 1e-8f)) {
                            min_dist_sq = d_sq;
                            best_closest = closest;
                            best_normal = make_float3(w * n.x, w * n.y, w * n.z);
                        } else if (fabsf(d_sq - min_dist_sq) <= 1e-5f * (min_dist_sq + 1e-8f)) {
                            best_normal = make_float3(best_normal.x + w * n.x, best_normal.y + w * n.y, best_normal.z + w * n.z);
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
        d_values[idx] = 3.402823466e+38f;
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
        if (dist > 2.5f * min_box_dim || dist < -2.5f * min_box_dim) {
            d_values[idx] = 3.402823466e+38f;
        } else {
            d_values[idx] = dist;
        }
    }
}

__global__ void init_exterior_border_kernel(int nx, int ny, int nz, float* d_values, uint8_t* d_exterior_mask) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_voxels = nx * ny * nz;
    if (idx >= total_voxels) return;

    int ix = idx % nx;
    int iy = (idx / nx) % ny;
    int iz = idx / (nx * ny);

    if (ix == 0 || ix == nx - 1 || iy == 0 || iy == ny - 1 || iz == 0 || iz == nz - 1) {
        if (d_values[idx] == 3.402823466e+38f || d_values[idx] > 0.0f) {
            if (d_values[idx] == 3.402823466e+38f) d_values[idx] = 10.0f;
            d_exterior_mask[idx] = 1;
        } else {
            d_exterior_mask[idx] = 0;
        }
    } else {
        if (d_values[idx] != 3.402823466e+38f && d_values[idx] > 0.0f) {
            d_exterior_mask[idx] = 1;
        } else {
            d_exterior_mask[idx] = 0;
        }
    }
}

__global__ void propagate_exterior_signs_kernel(int nx, int ny, int nz, float* d_values, uint8_t* d_exterior_mask, int* d_changed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_voxels = nx * ny * nz;
    if (idx >= total_voxels) return;

    if (d_exterior_mask[idx] == 1) return;
    if (d_values[idx] != 3.402823466e+38f && d_values[idx] <= 0.0f) return;

    int ix = idx % nx;
    int iy = (idx / nx) % ny;
    int iz = idx / (nx * ny);

    bool neighbor_exterior = false;
    if (ix > 0 && d_exterior_mask[idx - 1] == 1) neighbor_exterior = true;
    else if (ix < nx - 1 && d_exterior_mask[idx + 1] == 1) neighbor_exterior = true;
    else if (iy > 0 && d_exterior_mask[idx - nx] == 1) neighbor_exterior = true;
    else if (iy < ny - 1 && d_exterior_mask[idx + nx] == 1) neighbor_exterior = true;
    else if (iz > 0 && d_exterior_mask[idx - nx * ny] == 1) neighbor_exterior = true;
    else if (iz < nz - 1 && d_exterior_mask[idx + nx * ny] == 1) neighbor_exterior = true;

    if (neighbor_exterior) {
        if (d_values[idx] == 3.402823466e+38f) {
            d_values[idx] = 10.0f;
        }
        d_exterior_mask[idx] = 1;
        *d_changed = 1;
    }
}

__global__ void resolve_interior_cavities_kernel(int nx, int ny, int nz, float* d_values, const uint8_t* d_exterior_mask) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_voxels = nx * ny * nz;
    if (idx >= total_voxels) return;

    if (d_values[idx] == 3.402823466e+38f) {
        if (d_exterior_mask[idx] == 0) {
            d_values[idx] = -10.0f;
        } else {
            d_values[idx] = 10.0f;
        }
    }
}

__global__ void eikonal_relaxation_kernel(int nx, int ny, int nz, float vx, float vy, float vz, const float* d_values_in, float* d_values_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_voxels = nx * ny * nz;
    if (idx >= total_voxels) return;

    int ix = idx % nx;
    int iy = (idx / nx) % ny;
    int iz = idx / (nx * ny);

    float val = d_values_in[idx];
    if (fabsf(val) <= fminf(vx, fminf(vy, vz)) * 1.5f) {
        d_values_out[idx] = val;
        return;
    }

    float abs_val = fabsf(val);
    float sign_val = (val < 0.0f) ? -1.0f : 1.0f;

    if (ix > 0) abs_val = fminf(abs_val, fabsf(d_values_in[idx - 1]) + vx);
    if (ix < nx - 1) abs_val = fminf(abs_val, fabsf(d_values_in[idx + 1]) + vx);
    if (iy > 0) abs_val = fminf(abs_val, fabsf(d_values_in[idx - nx]) + vy);
    if (iy < ny - 1) abs_val = fminf(abs_val, fabsf(d_values_in[idx + nx]) + vy);
    if (iz > 0) abs_val = fminf(abs_val, fabsf(d_values_in[idx - nx * ny]) + vz);
    if (iz < nz - 1) abs_val = fminf(abs_val, fabsf(d_values_in[idx + nx * ny]) + vz);

    d_values_out[idx] = sign_val * abs_val;
}

__global__ void rasterize_triangles_to_voxels_kernel(
    const float* d_vertices, const int* d_faces, int num_faces,
    int vres, float ox, float oy, float oz, float vox_vx, float vox_vy, float vox_vz,
    uint8_t* d_voxels
) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_faces) return;

    int i0 = d_faces[3 * f + 0];
    int i1 = d_faces[3 * f + 1];
    int i2 = d_faces[3 * f + 2];

    float3 v0 = make_float3(d_vertices[3 * i0 + 0], d_vertices[3 * i0 + 1], d_vertices[3 * i0 + 2]);
    float3 v1 = make_float3(d_vertices[3 * i1 + 0], d_vertices[3 * i1 + 1], d_vertices[3 * i1 + 2]);
    float3 v2 = make_float3(d_vertices[3 * i2 + 0], d_vertices[3 * i2 + 1], d_vertices[3 * i2 + 2]);

    float min_x = fminf(v0.x, fminf(v1.x, v2.x));
    float min_y = fminf(v0.y, fminf(v1.y, v2.y));
    float min_z = fminf(v0.z, fminf(v1.z, v2.z));
    float max_x = fmaxf(v0.x, fmaxf(v1.x, v2.x));
    float max_y = fmaxf(v0.y, fmaxf(v1.y, v2.y));
    float max_z = fmaxf(v0.z, fmaxf(v1.z, v2.z));

    int bx0 = max(0, min(vres - 1, (int)floorf((min_x - ox) / vox_vx)));
    int bx1 = max(0, min(vres - 1, (int)floorf((max_x - ox) / vox_vx)));
    int by0 = max(0, min(vres - 1, (int)floorf((min_y - oy) / vox_vy)));
    int by1 = max(0, min(vres - 1, (int)floorf((max_y - oy) / vox_vy)));
    int bz0 = max(0, min(vres - 1, (int)floorf((min_z - oz) / vox_vz)));
    int bz1 = max(0, min(vres - 1, (int)floorf((max_z - oz) / vox_vz)));

    float max_v_dim = fmaxf(vox_vx, fmaxf(vox_vy, vox_vz)) * 0.8660254f;

    for (int bz = bz0; bz <= bz1; ++bz) {
        for (int by = by0; by <= by1; ++by) {
            for (int bx = bx0; bx <= bx1; ++bx) {
                float3 c = make_float3(ox + bx * vox_vx, oy + by * vox_vy, oz + bz * vox_vz);
                float3 closest;
                float d_sq = point_to_triangle_distance_sq(c, v0, v1, v2, closest);
                if (d_sq <= max_v_dim * max_v_dim) {
                    d_voxels[bx + (size_t)vres * (by + (size_t)vres * bz)] = 2;
                }
            }
        }
    }
}

__global__ void dilate_voxel_shell_kernel(int vres, const uint8_t* d_in, uint8_t* d_out, int r) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)vres * vres * vres;
    if (idx >= total) return;
    if (d_in[idx] == 2) {
        d_out[idx] = 2;
        return;
    }
    int ix = idx % vres;
    int iy = (idx / vres) % vres;
    int iz = idx / ((size_t)vres * vres);

    for (int dz = max(0, iz - r); dz <= min(vres - 1, iz + r); ++dz) {
        for (int dy = max(0, iy - r); dy <= min(vres - 1, iy + r); ++dy) {
            for (int dx = max(0, ix - r); dx <= min(vres - 1, ix + r); ++dx) {
                if (d_in[dx + (size_t)vres * (dy + (size_t)vres * dz)] == 2) {
                    d_out[idx] = 2;
                    return;
                }
            }
        }
    }
    d_out[idx] = d_in[idx];
}

__global__ void flood_fill_voxel_exterior_kernel(int vres, uint8_t* d_voxels, uint8_t* d_ext, int* d_changed) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)vres * vres * vres;
    if (idx >= total) return;

    if (d_ext[idx] == 1) return;
    if (d_voxels[idx] == 2) return;

    int ix = idx % vres;
    int iy = (idx / vres) % vres;
    int iz = idx / ((size_t)vres * vres);

    bool n_ext = false;
    if (ix > 0 && d_ext[idx - 1] == 1) n_ext = true;
    else if (ix < vres - 1 && d_ext[idx + 1] == 1) n_ext = true;
    else if (iy > 0 && d_ext[idx - vres] == 1) n_ext = true;
    else if (iy < vres - 1 && d_ext[idx + vres] == 1) n_ext = true;
    else if (iz > 0 && d_ext[idx - (size_t)vres * vres] == 1) n_ext = true;
    else if (iz < vres - 1 && d_ext[idx + (size_t)vres * vres] == 1) n_ext = true;

    if (n_ext) {
        d_ext[idx] = 1;
        *d_changed = 1;
    }
}

__global__ void init_voxel_exterior_kernel(int vres, uint8_t* d_voxels, uint8_t* d_ext) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)vres * vres * vres;
    if (idx >= total) return;
    int ix = idx % vres;
    int iy = (idx / vres) % vres;
    int iz = idx / ((size_t)vres * vres);

    if (ix == 0 || ix == vres - 1 || iy == 0 || iy == vres - 1 || iz == 0 || iz == vres - 1) {
        if (d_voxels[idx] != 2) d_ext[idx] = 1;
        else d_ext[idx] = 0;
    } else {
        d_ext[idx] = 0;
    }
}

__global__ void finalize_solid_voxels_kernel(int vres, uint8_t* d_voxels, const uint8_t* d_ext) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)vres * vres * vres;
    if (idx >= total) return;
    if (d_ext[idx] == 0) d_voxels[idx] = 1;
    else d_voxels[idx] = 0;
}

__global__ void init_voxel_labels_kernel(int vres, const uint8_t* d_voxels, int* d_labels) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)vres * vres * vres;
    if (idx >= total) return;
    if (d_voxels[idx] == 1) d_labels[idx] = (int)(idx % 2000000000);
    else d_labels[idx] = -1;
}

__global__ void propagate_voxel_labels_kernel(int vres, const uint8_t* d_voxels, int* d_labels, int* d_changed) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)vres * vres * vres;
    if (idx >= total) return;
    if (d_voxels[idx] != 1) return;

    int ix = idx % vres;
    int iy = (idx / vres) % vres;
    int iz = idx / ((size_t)vres * vres);

    int max_l = d_labels[idx];
    if (ix > 0 && d_voxels[idx - 1] == 1) max_l = max(max_l, d_labels[idx - 1]);
    if (ix < vres - 1 && d_voxels[idx + 1] == 1) max_l = max(max_l, d_labels[idx + 1]);
    if (iy > 0 && d_voxels[idx - vres] == 1) max_l = max(max_l, d_labels[idx - vres]);
    if (iy < vres - 1 && d_voxels[idx + vres] == 1) max_l = max(max_l, d_labels[idx + vres]);
    if (iz > 0 && d_voxels[idx - (size_t)vres * vres] == 1) max_l = max(max_l, d_labels[idx - (size_t)vres * vres]);
    if (iz < vres - 1 && d_voxels[idx + (size_t)vres * vres] == 1) max_l = max(max_l, d_labels[idx + (size_t)vres * vres]);

    if (max_l > d_labels[idx]) {
        d_labels[idx] = max_l;
        *d_changed = 1;
    }
}

__global__ void find_component_roots_kernel(int vres, const uint8_t* d_voxels, const int* d_labels, int* d_root_list, int* d_root_count, int max_roots) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)vres * vres * vres;
    if (idx >= total) return;
    if (d_voxels[idx] != 1) return;

    int l = d_labels[idx];
    if (l == (int)(idx % 2000000000)) {
        int pos = atomicAdd(d_root_count, 1);
        if (pos < max_roots) {
            d_root_list[pos] = l;
        }
    }
}

__global__ void count_root_sizes_kernel(int vres, const uint8_t* d_voxels, const int* d_labels, const int* d_root_list, int num_roots, int* d_root_counts) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)vres * vres * vres;
    if (idx >= total) return;
    if (d_voxels[idx] != 1) return;

    int l = d_labels[idx];
    for (int i = 0; i < num_roots; ++i) {
        if (d_root_list[i] == l) {
            atomicAdd(&d_root_counts[i], 1);
            break;
        }
    }
}

__global__ void keep_max_label_kernel(int vres, uint8_t* d_voxels, const int* d_labels, int max_label) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)vres * vres * vres;
    if (idx >= total) return;
    if (d_voxels[idx] == 1 && d_labels[idx] != max_label) {
        d_voxels[idx] = 0;
    }
}

__global__ void apply_solid_voxel_sign_mask_kernel(
    int nx, int ny, int nz, float ox, float oy, float oz, float vx, float vy, float vz,
    int vres, float vox_vx, float vox_vy, float vox_vz,
    const uint8_t* d_voxels, float* d_values
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_voxels = nx * ny * nz;
    if (idx >= total_voxels) return;

    int ix = idx % nx;
    int iy = (idx / nx) % ny;
    int iz = idx / (nx * ny);

    float3 p = make_float3(ox + ix * vx, oy + iy * vy, oz + iz * vz);

    int vbx = max(0, min(vres - 1, (int)floorf((p.x - ox) / vox_vx)));
    int vby = max(0, min(vres - 1, (int)floorf((p.y - oy) / vox_vy)));
    int vbz = max(0, min(vres - 1, (int)floorf((p.z - oz) / vox_vz)));
    size_t v_idx = vbx + (size_t)vres * (vby + (size_t)vres * vbz);

    float val = fabsf(d_values[idx]);
    if (d_voxels[v_idx] == 1) {
        d_values[idx] = -val;
    } else {
        d_values[idx] = val;
    }
}

__global__ void evaluate_sdf_from_solid_kernel(
    int nx, int ny, int nz, float ox, float oy, float oz, float vx, float vy, float vz,
    int vres, float vox_vx, float vox_vy, float vox_vz,
    const uint8_t* d_voxels, float* d_values
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_voxels = nx * ny * nz;
    if (idx >= total_voxels) return;

    int ix = idx % nx;
    int iy = (idx / nx) % ny;
    int iz = idx / (nx * ny);

    float3 p = make_float3(ox + ix * vx, oy + iy * vy, oz + iz * vz);

    int vbx = max(0, min(vres - 1, (int)floorf((p.x - ox) / vox_vx)));
    int vby = max(0, min(vres - 1, (int)floorf((p.y - oy) / vox_vy)));
    int vbz = max(0, min(vres - 1, (int)floorf((p.z - oz) / vox_vz)));
    size_t v_idx = vbx + (size_t)vres * (vby + (size_t)vres * vbz);
    float sign_val = (d_voxels[v_idx] == 1) ? -1.0f : 1.0f;

    int r = max(6, (int)ceilf(6.0f * (fmaxf(vx, fmaxf(vy, vz)) / fmaxf(vox_vx, fmaxf(vox_vy, vox_vz)))));
    r = min(16, r);

    float min_dist_sq = 3.402823466e+38f;

    for (int bz = max(0, vbz - r); bz <= min(vres - 1, vbz + r); ++bz) {
        for (int by = max(0, vby - r); by <= min(vres - 1, vby + r); ++by) {
            for (int bx = max(0, vbx - r); bx <= min(vres - 1, vbx + r); ++bx) {
                size_t c_idx = bx + (size_t)vres * (by + (size_t)vres * bz);
                if (d_voxels[c_idx] == 1) {
                    bool is_boundary = false;
                    if (bx == 0 || d_voxels[c_idx - 1] == 0) is_boundary = true;
                    else if (bx == vres - 1 || d_voxels[c_idx + 1] == 0) is_boundary = true;
                    else if (by == 0 || d_voxels[c_idx - vres] == 0) is_boundary = true;
                    else if (by == vres - 1 || d_voxels[c_idx + vres] == 0) is_boundary = true;
                    else if (bz == 0 || d_voxels[c_idx - (size_t)vres * vres] == 0) is_boundary = true;
                    else if (bz == vres - 1 || d_voxels[c_idx + (size_t)vres * vres] == 0) is_boundary = true;

                    if (is_boundary) {
                        float3 bc = make_float3(ox + bx * vox_vx, oy + by * vox_vy, oz + bz * vox_vz);
                        float3 diff = p - bc;
                        float d_sq = dot(diff, diff);
                        if (d_sq < min_dist_sq) {
                            min_dist_sq = d_sq;
                        }
                    }
                }
            }
        }
    }

    if (min_dist_sq == 3.402823466e+38f) {
        d_values[idx] = sign_val * 10.0f;
    } else {
        d_values[idx] = sign_val * sqrtf(min_dist_sq);
    }
}

DenseSdfGridDevice compute_mesh_sdf_device_cuda(
    const float* vertices, int num_vertices,
    const int* faces, int num_faces,
    int nx, int ny, int nz,
    float ox, float oy, float oz,
    float vx, float vy, float vz,
    float* out_ms,
    bool voxelize_first,
    int voxel_res
) {
    auto t0 = std::chrono::high_resolution_clock::now();

    float* d_vertices = nullptr;
    int* d_faces = nullptr;
    CUDA_CHECK_SDF(cudaMalloc(&d_vertices, num_vertices * 3 * sizeof(float)));
    CUDA_CHECK_SDF(cudaMalloc(&d_faces, num_faces * 3 * sizeof(int)));
    CUDA_CHECK_SDF(cudaMemcpy(d_vertices, vertices, num_vertices * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_SDF(cudaMemcpy(d_faces, faces, num_faces * 3 * sizeof(int), cudaMemcpyHostToDevice));

    float* d_values = nullptr;
    int total_voxels = nx * ny * nz;
    CUDA_CHECK_SDF(cudaMalloc(&d_values, total_voxels * sizeof(float)));
    int threads = 256;
    int blocks_v = (total_voxels + threads - 1) / threads;

    if (voxelize_first) {
        int vres = max(32, min(2048, voxel_res));
        int total_vox = vres * vres * vres;
        float vox_vx = (nx * vx) / vres;
        float vox_vy = (ny * vy) / vres;
        float vox_vz = (nz * vz) / vres;

        uint8_t* d_voxels = nullptr;
        uint8_t* d_ext = nullptr;
        int* d_changed = nullptr;
        CUDA_CHECK_SDF(cudaMalloc(&d_voxels, total_vox * sizeof(uint8_t)));
        CUDA_CHECK_SDF(cudaMalloc(&d_ext, total_vox * sizeof(uint8_t)));
        CUDA_CHECK_SDF(cudaMalloc(&d_changed, sizeof(int)));
        CUDA_CHECK_SDF(cudaMemset(d_voxels, 0, total_vox * sizeof(uint8_t)));

        int blocks_f = (num_faces + threads - 1) / threads;
        rasterize_triangles_to_voxels_kernel<<<blocks_f, threads>>>(d_vertices, d_faces, num_faces, vres, ox, oy, oz, vox_vx, vox_vy, vox_vz, d_voxels);
        CUDA_CHECK_SDF(cudaGetLastError());

        int blocks_vox = (total_vox + threads - 1) / threads;
        dilate_voxel_shell_kernel<<<blocks_vox, threads>>>(vres, d_voxels, d_ext, 3);
        CUDA_CHECK_SDF(cudaGetLastError());
        CUDA_CHECK_SDF(cudaMemcpy(d_voxels, d_ext, (size_t)total_vox * sizeof(uint8_t), cudaMemcpyDeviceToDevice));

        init_voxel_exterior_kernel<<<blocks_vox, threads>>>(vres, d_voxels, d_ext);
        CUDA_CHECK_SDF(cudaGetLastError());

        int max_iters = vres * 3;
        for (int iter = 0; iter < max_iters; ++iter) {
            int h_changed = 0;
            CUDA_CHECK_SDF(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));
            flood_fill_voxel_exterior_kernel<<<blocks_vox, threads>>>(vres, d_voxels, d_ext, d_changed);
            CUDA_CHECK_SDF(cudaGetLastError());
            CUDA_CHECK_SDF(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
            if (h_changed == 0) break;
        }

        finalize_solid_voxels_kernel<<<blocks_vox, threads>>>(vres, d_voxels, d_ext);
        CUDA_CHECK_SDF(cudaGetLastError());
        CUDA_CHECK_SDF(cudaFree(d_ext));
        d_ext = nullptr;

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

        blocks_f = (num_faces + threads - 1) / threads;
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

        evaluate_sdf_grid_kernel<<<blocks_v, threads>>>(nx, ny, nz, ox, oy, oz, vx, vy, vz, gx, gy, gz, bdx, bdy, bdz, d_bin_offsets, d_bin_triangles, d_vertices, d_faces, d_bin_active, d_values);
        CUDA_CHECK_SDF(cudaGetLastError());

        apply_solid_voxel_sign_mask_kernel<<<blocks_v, threads>>>(nx, ny, nz, ox, oy, oz, vx, vy, vz, vres, vox_vx, vox_vy, vox_vz, d_voxels, d_values);
        CUDA_CHECK_SDF(cudaGetLastError());

        if (d_bin_triangles) CUDA_CHECK_SDF(cudaFree(d_bin_triangles));
        CUDA_CHECK_SDF(cudaFree(d_bin_active));
        CUDA_CHECK_SDF(cudaFree(d_bin_counts));
        CUDA_CHECK_SDF(cudaFree(d_bin_offsets));
        CUDA_CHECK_SDF(cudaFree(d_bin_write_counts));
        CUDA_CHECK_SDF(cudaFree(d_changed));
        CUDA_CHECK_SDF(cudaFree(d_voxels));
    } else {
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

        evaluate_sdf_grid_kernel<<<blocks_v, threads>>>(nx, ny, nz, ox, oy, oz, vx, vy, vz, gx, gy, gz, bdx, bdy, bdz, d_bin_offsets, d_bin_triangles, d_vertices, d_faces, d_bin_active, d_values);
        CUDA_CHECK_SDF(cudaGetLastError());

        uint8_t* d_exterior_mask = nullptr;
        int* d_changed = nullptr;
        CUDA_CHECK_SDF(cudaMalloc(&d_exterior_mask, total_voxels * sizeof(uint8_t)));
        CUDA_CHECK_SDF(cudaMalloc(&d_changed, sizeof(int)));

        init_exterior_border_kernel<<<blocks_v, threads>>>(nx, ny, nz, d_values, d_exterior_mask);
        CUDA_CHECK_SDF(cudaGetLastError());

        for (int iter = 0; iter < 40; ++iter) {
            int h_changed = 0;
            CUDA_CHECK_SDF(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));
            propagate_exterior_signs_kernel<<<blocks_v, threads>>>(nx, ny, nz, d_values, d_exterior_mask, d_changed);
            CUDA_CHECK_SDF(cudaGetLastError());
            CUDA_CHECK_SDF(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
            if (h_changed == 0) break;
        }

        resolve_interior_cavities_kernel<<<blocks_v, threads>>>(nx, ny, nz, d_values, d_exterior_mask);
        CUDA_CHECK_SDF(cudaGetLastError());

        float* d_values_tmp = nullptr;
        CUDA_CHECK_SDF(cudaMalloc(&d_values_tmp, total_voxels * sizeof(float)));
        for (int iter = 0; iter < 4; ++iter) {
            if (iter % 2 == 0) {
                eikonal_relaxation_kernel<<<blocks_v, threads>>>(nx, ny, nz, vx, vy, vz, d_values, d_values_tmp);
            } else {
                eikonal_relaxation_kernel<<<blocks_v, threads>>>(nx, ny, nz, vx, vy, vz, d_values_tmp, d_values);
            }
            CUDA_CHECK_SDF(cudaGetLastError());
        }

        CUDA_CHECK_SDF(cudaFree(d_values_tmp));
        CUDA_CHECK_SDF(cudaFree(d_changed));
        CUDA_CHECK_SDF(cudaFree(d_exterior_mask));
        CUDA_CHECK_SDF(cudaDeviceSynchronize());

        CUDA_CHECK_SDF(cudaFree(d_bin_active));
        if (d_bin_triangles) CUDA_CHECK_SDF(cudaFree(d_bin_triangles));
        CUDA_CHECK_SDF(cudaFree(d_bin_write_counts));
        CUDA_CHECK_SDF(cudaFree(d_bin_offsets));
        CUDA_CHECK_SDF(cudaFree(d_bin_counts));
    }

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
    float* out_ms,
    bool voxelize_first,
    int voxel_res
) {
    DenseSdfGridDevice dev_grid = compute_mesh_sdf_device_cuda(vertices, num_vertices, faces, num_faces, nx, ny, nz, ox, oy, oz, vx, vy, vz, out_ms, voxelize_first, voxel_res);

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
