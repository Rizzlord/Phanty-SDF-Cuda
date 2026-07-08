#include "dc_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>

#define CUDA_CHECK(val) { cuda_check_err((val), #val, __FILE__, __LINE__); }
inline void cuda_check_err(cudaError_t err, const char* const func, const char* const file, int const line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << err << "(" << cudaGetErrorString(err) << ") \"" << func << "\"\n";
        exit(EXIT_FAILURE);
    }
}

struct GridParams {
    int nx, ny, nz;
    float vx, vy, vz;
    float ox, oy, oz;
};

struct DeviceCounters {
    int qef_fallbacks;
    int out_of_cell_clamps;
};

__device__ __host__ inline int grid_index(int ix, int iy, int iz, int nx, int ny, int nz) {
    return ix + nx * (iy + ny * iz);
}

__device__ __host__ inline int cell_index(int ix, int iy, int iz, int nx, int ny, int nz) {
    return ix + (nx - 1) * (iy + (ny - 1) * iz);
}

__device__ inline float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ inline float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ inline float3 operator*(float a, float3 b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}
__device__ inline float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ inline float3 cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ inline float3 gradient_trilinear(
    const float corner_sdf[8],
    const float3 corner_pos[8],
    float3 p
) {
    float S000 = corner_sdf[0];
    float S100 = corner_sdf[1];
    float S110 = corner_sdf[2];
    float S010 = corner_sdf[3];
    float S001 = corner_sdf[4];
    float S101 = corner_sdf[5];
    float S111 = corner_sdf[6];
    float S011 = corner_sdf[7];

    float x0 = corner_pos[0].x;
    float x1 = corner_pos[1].x;
    float y0 = corner_pos[0].y;
    float y1 = corner_pos[3].y;
    float z0 = corner_pos[0].z;
    float z1 = corner_pos[4].z;

    float dx = x1 - x0;
    float dy = y1 - y0;
    float dz = z1 - z0;

    float u = 0.5f;
    float v = 0.5f;
    float w = 0.5f;

    if (fabsf(dx) > 1e-12f) u = (p.x - x0) / dx;
    if (fabsf(dy) > 1e-12f) v = (p.y - y0) / dy;
    if (fabsf(dz) > 1e-12f) w = (p.z - z0) / dz;

    u = fmaxf(0.0f, fminf(1.0f, u));
    v = fmaxf(0.0f, fminf(1.0f, v));
    w = fmaxf(0.0f, fminf(1.0f, w));

    float omu = 1.0f - u;
    float omv = 1.0f - v;
    float omw = 1.0f - w;

    float df_du = (S100 - S000) * omv * omw
                + (S110 - S010) * v * omw
                + (S101 - S001) * omv * w
                + (S111 - S011) * v * w;

    float df_dv = (S010 - S000) * omu * omw
                + (S110 - S100) * u * omw
                + (S011 - S001) * omu * w
                + (S111 - S101) * u * w;

    float df_dw = (S001 - S000) * omu * omv
                + (S101 - S100) * u * omv
                + (S011 - S010) * omu * v
                + (S111 - S110) * u * v;

    float3 g = make_float3(0.0f, 0.0f, 0.0f);
    if (fabsf(dx) > 1e-12f) g.x = df_du / dx;
    if (fabsf(dy) > 1e-12f) g.y = df_dv / dy;
    if (fabsf(dz) > 1e-12f) g.z = df_dw / dz;

    return g;
}

__device__ inline float3 get_gradient(const float* sdf, int gi, int gj, int gk, GridParams params) {
    int im = (gi > 0) ? gi - 1 : 0;
    int ip = (gi < params.nx - 1) ? gi + 1 : params.nx - 1;
    int jm = (gj > 0) ? gj - 1 : 0;
    int jp = (gj < params.ny - 1) ? gj + 1 : params.ny - 1;
    int km = (gk > 0) ? gk - 1 : 0;
    int kp = (gk < params.nz - 1) ? gk + 1 : params.nz - 1;

    float s_im = sdf[grid_index(im, gj, gk, params.nx, params.ny, params.nz)];
    float s_ip = sdf[grid_index(ip, gj, gk, params.nx, params.ny, params.nz)];
    float s_jm = sdf[grid_index(gi, jm, gk, params.nx, params.ny, params.nz)];
    float s_jp = sdf[grid_index(gi, jp, gk, params.nx, params.ny, params.nz)];
    float s_km = sdf[grid_index(gi, gj, km, params.nx, params.ny, params.nz)];
    float s_kp = sdf[grid_index(gi, gj, kp, params.nx, params.ny, params.nz)];

    float3 g = make_float3(0.0f, 0.0f, 0.0f);
    float dx = (ip - im) * params.vx;
    float dy = (jp - jm) * params.vy;
    float dz = (kp - km) * params.vz;

    if (dx > 1e-12f) g.x = (s_ip - s_im) / dx;
    if (dy > 1e-12f) g.y = (s_jp - s_jm) / dy;
    if (dz > 1e-12f) g.z = (s_kp - s_km) / dz;
    return g;
}

__device__ void jacobi_solve_3x3(
    const float B[9],
    const float d[3],
    float svd_threshold,
    float y[3],
    bool& singular
) {
    float V[9] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };
    float A_mat[9];
    for (int i = 0; i < 9; ++i) A_mat[i] = B[i];

    const int max_iters = 15;
    for (int iter = 0; iter < max_iters; ++iter) {
        int p = 0, q = 1;
        float max_val = fabsf(A_mat[1]);
        if (fabsf(A_mat[2]) > max_val) {
            p = 0; q = 2;
            max_val = fabsf(A_mat[2]);
        }
        if (fabsf(A_mat[5]) > max_val) {
            p = 1; q = 2;
            max_val = fabsf(A_mat[5]);
        }

        if (max_val < 1e-7f) {
            break;
        }

        float ap = A_mat[p * 3 + p];
        float aq = A_mat[q * 3 + q];
        float apq = A_mat[p * 3 + q];

        float phi = 0.5f * (aq - ap) / apq;
        float t = 1.0f / (fabsf(phi) + sqrtf(phi * phi + 1.0f));
        if (phi < 0.0f) t = -t;
        float c = 1.0f / sqrtf(t * t + 1.0f);
        float s = t * c;

        for (int i = 0; i < 3; ++i) {
            if (i != p && i != q) {
                float aip = A_mat[i * 3 + p];
                float aiq = A_mat[i * 3 + q];
                A_mat[i * 3 + p] = A_mat[p * 3 + i] = c * aip - s * aiq;
                A_mat[i * 3 + q] = A_mat[q * 3 + i] = s * aip + c * aiq;
            }
        }
        A_mat[p * 3 + p] = ap - t * apq;
        A_mat[q * 3 + q] = aq + t * apq;
        A_mat[p * 3 + q] = A_mat[q * 3 + p] = 0.0f;

        for (int i = 0; i < 3; ++i) {
            float vip = V[i * 3 + p];
            float viq = V[i * 3 + q];
            V[i * 3 + p] = c * vip - s * viq;
            V[i * 3 + q] = s * vip + c * viq;
        }
    }

    float S[3] = { A_mat[0], A_mat[4], A_mat[8] };
    float S_inv[3];
    int ok_count = 0;
    for (int i = 0; i < 3; ++i) {
        if (S[i] > svd_threshold) {
            S_inv[i] = 1.0f / S[i];
            ok_count++;
        } else {
            S_inv[i] = 0.0f;
        }
    }

    singular = (ok_count == 0);

    float tmp[3];
    tmp[0] = V[0] * d[0] + V[3] * d[1] + V[6] * d[2];
    tmp[1] = V[1] * d[0] + V[4] * d[1] + V[7] * d[2];
    tmp[2] = V[2] * d[0] + V[5] * d[1] + V[8] * d[2];

    tmp[0] *= S_inv[0];
    tmp[1] *= S_inv[1];
    tmp[2] *= S_inv[2];

    y[0] = V[0] * tmp[0] + V[1] * tmp[1] + V[2] * tmp[2];
    y[1] = V[3] * tmp[0] + V[4] * tmp[1] + V[5] * tmp[2];
    y[2] = V[6] * tmp[0] + V[7] * tmp[1] + V[8] * tmp[2];
}

__global__ void mark_active_cells_kernel(
    const float* d_sdf,
    uint8_t* d_active_flags,
    int cell_count,
    GridParams params
) {
    int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_idx >= cell_count) return;

    int cx = params.nx - 1;
    int cy = params.ny - 1;
    int ix = cell_idx % cx;
    int iy = (cell_idx / cx) % cy;
    int iz = cell_idx / (cx * cy);

    const int corner_offsets[8][3] = {
        {0, 0, 0},
        {1, 0, 0},
        {1, 1, 0},
        {0, 1, 0},
        {0, 0, 1},
        {1, 0, 1},
        {1, 1, 1},
        {0, 1, 1}
    };

    bool has_neg = false;
    bool has_pos = false;

    for (int c = 0; c < 8; ++c) {
        int g_ix = ix + corner_offsets[c][0];
        int g_iy = iy + corner_offsets[c][1];
        int g_iz = iz + corner_offsets[c][2];
        int g_idx = grid_index(g_ix, g_iy, g_iz, params.nx, params.ny, params.nz);
        float val = d_sdf[g_idx];
        if (val < 0.0f) has_neg = true;
        if (val > 0.0f) has_pos = true;
    }

    d_active_flags[cell_idx] = (has_neg && has_pos) ? 1 : 0;
}

__global__ void compute_vertices_kernel(
    const int* d_active_cell_indices,
    const float* d_sdf,
    float3* d_vertices,
    int* d_cell_to_vertex_map,
    int active_cell_count,
    GridParams params,
    DeviceCounters* d_counters
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= active_cell_count) return;

    int cell_idx = d_active_cell_indices[idx];
    d_cell_to_vertex_map[cell_idx] = idx;

    int cx = params.nx - 1;
    int cy = params.ny - 1;
    int ix = cell_idx % cx;
    int iy = (cell_idx / cx) % cy;
    int iz = cell_idx / (cx * cy);

    const int corner_offsets[8][3] = {
        {0, 0, 0},
        {1, 0, 0},
        {1, 1, 0},
        {0, 1, 0},
        {0, 0, 1},
        {1, 0, 1},
        {1, 1, 1},
        {0, 1, 1}
    };

    const int edge_pairs[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7}
    };

    float3 pts[12];
    float3 normals[12];
    float3 centroid = make_float3(0.0f, 0.0f, 0.0f);
    int m = 0;

    for (int e = 0; e < 12; ++e) {
        int a = edge_pairs[e][0];
        int b = edge_pairs[e][1];

        int a_ix = ix + corner_offsets[a][0];
        int a_iy = iy + corner_offsets[a][1];
        int a_iz = iz + corner_offsets[a][2];

        int b_ix = ix + corner_offsets[b][0];
        int b_iy = iy + corner_offsets[b][1];
        int b_iz = iz + corner_offsets[b][2];

        float fa = d_sdf[grid_index(a_ix, a_iy, a_iz, params.nx, params.ny, params.nz)];
        float fb = d_sdf[grid_index(b_ix, b_iy, b_iz, params.nx, params.ny, params.nz)];

        if (fa * fb <= 0.0f) {
            float t = fa / (fa - fb + 1e-12f);
            t = fmaxf(0.0f, fminf(1.0f, t));

            float3 pa = make_float3(params.ox + a_ix * params.vx, params.oy + a_iy * params.vy, params.oz + a_iz * params.vz);
            float3 pb = make_float3(params.ox + b_ix * params.vx, params.oy + b_iy * params.vy, params.oz + b_iz * params.vz);
            float3 p = (1.0f - t) * pa + t * pb;

            float corner_sdf_vals[8];
            float3 corner_pos_vals[8];
            for (int c = 0; c < 8; ++c) {
                int c_ix = ix + corner_offsets[c][0];
                int c_iy = iy + corner_offsets[c][1];
                int c_iz = iz + corner_offsets[c][2];
                corner_sdf_vals[c] = d_sdf[grid_index(c_ix, c_iy, c_iz, params.nx, params.ny, params.nz)];
                corner_pos_vals[c] = make_float3(params.ox + c_ix * params.vx, params.oy + c_iy * params.vy, params.oz + c_iz * params.vz);
            }

            float3 n = gradient_trilinear(corner_sdf_vals, corner_pos_vals, p);
            float n_len = sqrtf(dot(n, n));
            if (n_len > 1e-6f) {
                n = (1.0f / n_len) * n;
            }

            pts[m] = p;
            normals[m] = n;
            centroid = centroid + p;
            m++;
        }
    }

    if (m == 0) {
        float3 cell_min = make_float3(params.ox + ix * params.vx, params.oy + iy * params.vy, params.oz + iz * params.vz);
        d_vertices[idx] = cell_min + 0.5f * make_float3(params.vx, params.vy, params.vz);
        return;
    }

    centroid = (1.0f / (float)m) * centroid;

    float B[9] = {0.0f};
    float d[3] = {0.0f};
    for (int i = 0; i < m; ++i) {
        float3 n = normals[i];
        float3 diff = pts[i] - centroid;
        float dot_val = dot(n, diff);

        B[0] += n.x * n.x; B[1] += n.x * n.y; B[2] += n.x * n.z;
        B[3] += n.y * n.x; B[4] += n.y * n.y; B[5] += n.y * n.z;
        B[6] += n.z * n.x; B[7] += n.z * n.y; B[8] += n.z * n.z;

        d[0] += n.x * dot_val;
        d[1] += n.y * dot_val;
        d[2] += n.z * dot_val;
    }

    float y[3] = {0.0f};
    bool singular = false;
    jacobi_solve_3x3(B, d, 0.01f, y, singular);

    float3 vertex = centroid + make_float3(y[0], y[1], y[2]);

    bool has_nan_inf = isnan(vertex.x) || isnan(vertex.y) || isnan(vertex.z) ||
                       isinf(vertex.x) || isinf(vertex.y) || isinf(vertex.z);

    if (singular || has_nan_inf) {
        atomicAdd(&d_counters->qef_fallbacks, 1);
        vertex = centroid;
    }

    float3 cell_min = make_float3(params.ox + ix * params.vx, params.oy + iy * params.vy, params.oz + iz * params.vz);
    float3 cell_max = cell_min + make_float3(params.vx, params.vy, params.vz);

    if (vertex.x < cell_min.x || vertex.x > cell_max.x ||
        vertex.y < cell_min.y || vertex.y > cell_max.y ||
        vertex.z < cell_min.z || vertex.z > cell_max.z) {
        atomicAdd(&d_counters->out_of_cell_clamps, 1);
        vertex.x = fmaxf(cell_min.x, fminf(cell_max.x, vertex.x));
        vertex.y = fmaxf(cell_min.y, fminf(cell_max.y, vertex.y));
        vertex.z = fmaxf(cell_min.z, fminf(cell_max.z, vertex.z));
    }

    d_vertices[idx] = vertex;
}

__global__ void count_faces_kernel(
    const float* d_sdf,
    const int* d_cell_to_vertex_map,
    int* d_edge_face_counts,
    int num_x_edges,
    int num_y_edges,
    int num_z_edges,
    int total_edges,
    GridParams params
) {
    int global_edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_edge_idx >= total_edges) return;

    int cx = params.nx - 1;
    int cy = params.ny - 1;
    int cz = params.nz - 1;

    int gv0_x = 0, gv0_y = 0, gv0_z = 0;
    int gv1_x = 0, gv1_y = 0, gv1_z = 0;
    int c0 = -1, c1 = -1, c2 = -1, c3 = -1;

    if (global_edge_idx < num_x_edges) {
        int idx = global_edge_idx;
        int i = idx % cx;
        int j = 1 + (idx / cx) % (cy - 1);
        int k = 1 + idx / (cx * (cy - 1));

        gv0_x = i;   gv0_y = j; gv0_z = k;
        gv1_x = i+1; gv1_y = j; gv1_z = k;

        c0 = cell_index(i, j-1, k-1, params.nx, params.ny, params.nz);
        c1 = cell_index(i, j,   k-1, params.nx, params.ny, params.nz);
        c2 = cell_index(i, j,   k,   params.nx, params.ny, params.nz);
        c3 = cell_index(i, j-1, k,   params.nx, params.ny, params.nz);
    } else if (global_edge_idx < num_x_edges + num_y_edges) {
        int idx = global_edge_idx - num_x_edges;
        int i = 1 + idx % (cx - 1);
        int j = (idx / (cx - 1)) % cy;
        int k = 1 + idx / ((cx - 1) * cy);

        gv0_x = i; gv0_y = j;   gv0_z = k;
        gv1_x = i; gv1_y = j+1; gv1_z = k;

        c0 = cell_index(i-1, j, k-1, params.nx, params.ny, params.nz);
        c1 = cell_index(i,   j, k-1, params.nx, params.ny, params.nz);
        c2 = cell_index(i,   j, k,   params.nx, params.ny, params.nz);
        c3 = cell_index(i-1, j, k,   params.nx, params.ny, params.nz);
    } else {
        int idx = global_edge_idx - num_x_edges - num_y_edges;
        int i = 1 + idx % (cx - 1);
        int j = 1 + (idx / (cx - 1)) % (cy - 1);
        int k = idx / ((cx - 1) * (cy - 1));

        gv0_x = i; gv0_y = j; gv0_z = k;
        gv1_x = i; gv1_y = j; gv1_z = k+1;

        c0 = cell_index(i-1, j-1, k, params.nx, params.ny, params.nz);
        c1 = cell_index(i,   j-1, k, params.nx, params.ny, params.nz);
        c2 = cell_index(i,   j,   k, params.nx, params.ny, params.nz);
        c3 = cell_index(i-1, j,   k, params.nx, params.ny, params.nz);
    }

    int v0 = d_cell_to_vertex_map[c0];
    int v1 = d_cell_to_vertex_map[c1];
    int v2 = d_cell_to_vertex_map[c2];
    int v3 = d_cell_to_vertex_map[c3];

    bool valid = false;
    if (v0 >= 0 && v1 >= 0 && v2 >= 0 && v3 >= 0) {
        int gv0_idx = grid_index(gv0_x, gv0_y, gv0_z, params.nx, params.ny, params.nz);
        int gv1_idx = grid_index(gv1_x, gv1_y, gv1_z, params.nx, params.ny, params.nz);
        float s0 = d_sdf[gv0_idx];
        float s1 = d_sdf[gv1_idx];
        if (s0 * s1 < 0.0f) {
            valid = true;
        }
    }

    d_edge_face_counts[global_edge_idx] = valid ? 1 : 0;
}

__global__ void emit_faces_kernel(
    const float* d_sdf,
    const int* d_cell_to_vertex_map,
    const float3* d_vertices,
    const int* d_edge_face_counts,
    const int* d_scanned_edge_face_counts,
    int* d_faces,
    int num_x_edges,
    int num_y_edges,
    int num_z_edges,
    int total_edges,
    GridParams params
) {
    int global_edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_edge_idx >= total_edges) return;

    if (d_edge_face_counts[global_edge_idx] == 0) return;
    int out_idx = d_scanned_edge_face_counts[global_edge_idx];

    int cx = params.nx - 1;
    int cy = params.ny - 1;
    int cz = params.nz - 1;

    int gv0_x = 0, gv0_y = 0, gv0_z = 0;
    int gv1_x = 0, gv1_y = 0, gv1_z = 0;
    int c0 = -1, c1 = -1, c2 = -1, c3 = -1;

    if (global_edge_idx < num_x_edges) {
        int idx = global_edge_idx;
        int i = idx % cx;
        int j = 1 + (idx / cx) % (cy - 1);
        int k = 1 + idx / (cx * (cy - 1));

        gv0_x = i;   gv0_y = j; gv0_z = k;
        gv1_x = i+1; gv1_y = j; gv1_z = k;

        c0 = cell_index(i, j-1, k-1, params.nx, params.ny, params.nz);
        c1 = cell_index(i, j,   k-1, params.nx, params.ny, params.nz);
        c2 = cell_index(i, j,   k,   params.nx, params.ny, params.nz);
        c3 = cell_index(i, j-1, k,   params.nx, params.ny, params.nz);
    } else if (global_edge_idx < num_x_edges + num_y_edges) {
        int idx = global_edge_idx - num_x_edges;
        int i = 1 + idx % (cx - 1);
        int j = (idx / (cx - 1)) % cy;
        int k = 1 + idx / ((cx - 1) * cy);

        gv0_x = i; gv0_y = j;   gv0_z = k;
        gv1_x = i; gv1_y = j+1; gv1_z = k;

        c0 = cell_index(i-1, j, k-1, params.nx, params.ny, params.nz);
        c1 = cell_index(i,   j, k-1, params.nx, params.ny, params.nz);
        c2 = cell_index(i,   j, k,   params.nx, params.ny, params.nz);
        c3 = cell_index(i-1, j, k,   params.nx, params.ny, params.nz);
    } else {
        int idx = global_edge_idx - num_x_edges - num_y_edges;
        int i = 1 + idx % (cx - 1);
        int j = 1 + (idx / (cx - 1)) % (cy - 1);
        int k = idx / ((cx - 1) * (cy - 1));

        gv0_x = i; gv0_y = j; gv0_z = k;
        gv1_x = i; gv1_y = j; gv1_z = k+1;

        c0 = cell_index(i-1, j-1, k, params.nx, params.ny, params.nz);
        c1 = cell_index(i,   j-1, k, params.nx, params.ny, params.nz);
        c2 = cell_index(i,   j,   k, params.nx, params.ny, params.nz);
        c3 = cell_index(i-1, j,   k, params.nx, params.ny, params.nz);
    }

    int v0 = d_cell_to_vertex_map[c0];
    int v1 = d_cell_to_vertex_map[c1];
    int v2 = d_cell_to_vertex_map[c2];
    int v3 = d_cell_to_vertex_map[c3];

    float3 pa = d_vertices[v0];
    float3 pb = d_vertices[v1];
    float3 pc = d_vertices[v2];
    float3 pd = d_vertices[v3];

    float3 quad_n = cross(pb - pa, pd - pa);
    if (dot(quad_n, quad_n) < 1e-12f) {
        quad_n = cross(pc - pa, pd - pa);
    }

    float3 g0 = get_gradient(d_sdf, gv0_x, gv0_y, gv0_z, params);
    float3 g1 = get_gradient(d_sdf, gv1_x, gv1_y, gv1_z, params);
    float3 grid_n = g0 + g1;

    if (dot(grid_n, grid_n) > 0.0f && dot(quad_n, quad_n) > 0.0f) {
        if (dot(quad_n, grid_n) < 0.0f) {
            int tmp = v1;
            v1 = v3;
            v3 = tmp;
        }
    }

    d_faces[4 * out_idx + 0] = v0;
    d_faces[4 * out_idx + 1] = v1;
    d_faces[4 * out_idx + 2] = v2;
    d_faces[4 * out_idx + 3] = v3;
}

struct is_active_op {
    const uint8_t* active_flags;
    __host__ __device__ bool operator()(int cell_idx) const {
        return active_flags[cell_idx] != 0;
    }
};

DualContouringMesh CudaDualContouringBackend::extract(const DenseSdfGrid& grid, DualContouringStats& stats) {
    cudaEvent_t event_start, event_stop;
    CUDA_CHECK(cudaEventCreate(&event_start));
    CUDA_CHECK(cudaEventCreate(&event_stop));

    float upload_ms = 0.0f;
    int total_vertices = grid.nx * grid.ny * grid.nz;
    float* d_sdf = nullptr;

    CUDA_CHECK(cudaEventRecord(event_start));
    CUDA_CHECK(cudaMalloc(&d_sdf, total_vertices * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_sdf, grid.values.data(), total_vertices * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(event_stop));
    CUDA_CHECK(cudaEventSynchronize(event_stop));
    CUDA_CHECK(cudaEventElapsedTime(&upload_ms, event_start, event_stop));

    DenseSdfGridDevice dev_grid;
    dev_grid.nx = grid.nx;
    dev_grid.ny = grid.ny;
    dev_grid.nz = grid.nz;
    dev_grid.vx = grid.vx;
    dev_grid.vy = grid.vy;
    dev_grid.vz = grid.vz;
    dev_grid.ox = grid.ox;
    dev_grid.oy = grid.oy;
    dev_grid.oz = grid.oz;
    dev_grid.d_values = d_sdf;

    DualContouringMesh mesh = extract_device(dev_grid, stats);
    stats.upload_ms += upload_ms;

    CUDA_CHECK(cudaFree(d_sdf));
    CUDA_CHECK(cudaEventDestroy(event_start));
    CUDA_CHECK(cudaEventDestroy(event_stop));
    return mesh;
}

DualContouringMesh CudaDualContouringBackend::extract_device(const DenseSdfGridDevice& grid, DualContouringStats& stats) {
    cudaEvent_t event_start, event_stop;
    CUDA_CHECK(cudaEventCreate(&event_start));
    CUDA_CHECK(cudaEventCreate(&event_stop));

    float upload_ms = 0.0f;
    float marking_ms = 0.0f;
    float compaction_ms = 0.0f;
    float qef_ms = 0.0f;
    float face_emission_ms = 0.0f;
    float download_ms = 0.0f;

    auto time_start_all = std::chrono::high_resolution_clock::now();

    int total_cells = (grid.nx - 1) * (grid.ny - 1) * (grid.nz - 1);
    int total_vertices = grid.nx * grid.ny * grid.nz;

    GridParams params;
    params.nx = grid.nx;
    params.ny = grid.ny;
    params.nz = grid.nz;
    params.vx = grid.vx;
    params.vy = grid.vy;
    params.vz = grid.vz;
    params.ox = grid.ox;
    params.oy = grid.oy;
    params.oz = grid.oz;

    float* d_sdf = (float*)grid.d_values;
    uint8_t* d_active_flags = nullptr;
    int* d_active_cell_indices = nullptr;
    int* d_cell_to_vertex_map = nullptr;
    float3* d_vertices = nullptr;
    DeviceCounters* d_counters = nullptr;
    CUDA_CHECK(cudaMalloc(&d_active_flags, total_cells * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_active_cell_indices, total_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_to_vertex_map, total_cells * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_cell_to_vertex_map, -1, total_cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counters, sizeof(DeviceCounters)));
    CUDA_CHECK(cudaMemset(d_counters, 0, sizeof(DeviceCounters)));

    int block_size = 256;
    int grid_size_cells = (total_cells + block_size - 1) / block_size;

    CUDA_CHECK(cudaEventRecord(event_start));
    mark_active_cells_kernel<<<grid_size_cells, block_size>>>(d_sdf, d_active_flags, total_cells, params);
    CUDA_CHECK(cudaEventRecord(event_stop));
    CUDA_CHECK(cudaEventSynchronize(event_stop));
    CUDA_CHECK(cudaEventElapsedTime(&marking_ms, event_start, event_stop));

    CUDA_CHECK(cudaEventRecord(event_start));
    thrust::counting_iterator<int> count_begin(0);
    thrust::counting_iterator<int> count_end(total_cells);
    is_active_op active_pred{ d_active_flags };
    int* new_end = thrust::copy_if(
        thrust::device,
        count_begin,
        count_end,
        d_active_cell_indices,
        active_pred
    );
    int active_cell_count = new_end - d_active_cell_indices;
    CUDA_CHECK(cudaEventRecord(event_stop));
    CUDA_CHECK(cudaEventSynchronize(event_stop));
    CUDA_CHECK(cudaEventElapsedTime(&compaction_ms, event_start, event_stop));

    DualContouringMesh mesh;

    if (active_cell_count > 0) {
        CUDA_CHECK(cudaMalloc(&d_vertices, active_cell_count * sizeof(float3)));

        int grid_size_active = (active_cell_count + block_size - 1) / block_size;

        CUDA_CHECK(cudaEventRecord(event_start));
        compute_vertices_kernel<<<grid_size_active, block_size>>>(
            d_active_cell_indices,
            d_sdf,
            d_vertices,
            d_cell_to_vertex_map,
            active_cell_count,
            params,
            d_counters
        );
        CUDA_CHECK(cudaEventRecord(event_stop));
        CUDA_CHECK(cudaEventSynchronize(event_stop));
        CUDA_CHECK(cudaEventElapsedTime(&qef_ms, event_start, event_stop));

        int cx = grid.nx - 1;
        int cy = grid.ny - 1;
        int cz = grid.nz - 1;

        int num_x_edges = cx * (cy - 1) * (cz - 1);
        int num_y_edges = (cx - 1) * cy * (cz - 1);
        int num_z_edges = (cx - 1) * (cy - 1) * cz;
        int total_edges = num_x_edges + num_y_edges + num_z_edges;

        int* d_edge_face_counts = nullptr;
        int* d_scanned_edge_face_counts = nullptr;
        int* d_faces = nullptr;

        CUDA_CHECK(cudaMalloc(&d_edge_face_counts, total_edges * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_scanned_edge_face_counts, total_edges * sizeof(int)));

        int grid_size_edges = (total_edges + block_size - 1) / block_size;

        CUDA_CHECK(cudaEventRecord(event_start));
        count_faces_kernel<<<grid_size_edges, block_size>>>(
            d_sdf,
            d_cell_to_vertex_map,
            d_edge_face_counts,
            num_x_edges,
            num_y_edges,
            num_z_edges,
            total_edges,
            params
        );

        thrust::exclusive_scan(thrust::device, d_edge_face_counts, d_edge_face_counts + total_edges, d_scanned_edge_face_counts);

        int last_count = 0;
        int last_scanned = 0;
        CUDA_CHECK(cudaMemcpy(&last_count, d_edge_face_counts + total_edges - 1, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&last_scanned, d_scanned_edge_face_counts + total_edges - 1, sizeof(int), cudaMemcpyDeviceToHost));
        int total_faces = last_scanned + last_count;

        if (total_faces > 0) {
            CUDA_CHECK(cudaMalloc(&d_faces, total_faces * 4 * sizeof(int)));
            emit_faces_kernel<<<grid_size_edges, block_size>>>(
                d_sdf,
                d_cell_to_vertex_map,
                d_vertices,
                d_edge_face_counts,
                d_scanned_edge_face_counts,
                d_faces,
                num_x_edges,
                num_y_edges,
                num_z_edges,
                total_edges,
                params
            );
        }
        CUDA_CHECK(cudaEventRecord(event_stop));
        CUDA_CHECK(cudaEventSynchronize(event_stop));
        CUDA_CHECK(cudaEventElapsedTime(&face_emission_ms, event_start, event_stop));

        CUDA_CHECK(cudaEventRecord(event_start));
        mesh.vertices.resize(active_cell_count * 3);
        CUDA_CHECK(cudaMemcpy(mesh.vertices.data(), d_vertices, active_cell_count * sizeof(float3), cudaMemcpyDeviceToHost));

        if (total_faces > 0) {
            mesh.faces.resize(total_faces * 4);
            CUDA_CHECK(cudaMemcpy(mesh.faces.data(), d_faces, total_faces * 4 * sizeof(int), cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaEventRecord(event_stop));
        CUDA_CHECK(cudaEventSynchronize(event_stop));
        CUDA_CHECK(cudaEventElapsedTime(&download_ms, event_start, event_stop));

        if (d_faces) CUDA_CHECK(cudaFree(d_faces));
        CUDA_CHECK(cudaFree(d_edge_face_counts));
        CUDA_CHECK(cudaFree(d_scanned_edge_face_counts));
        CUDA_CHECK(cudaFree(d_vertices));
    }

    DeviceCounters host_counters;
    CUDA_CHECK(cudaMemcpy(&host_counters, d_counters, sizeof(DeviceCounters), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_active_flags));
    CUDA_CHECK(cudaFree(d_active_cell_indices));
    CUDA_CHECK(cudaFree(d_cell_to_vertex_map));
    CUDA_CHECK(cudaFree(d_counters));

    CUDA_CHECK(cudaEventDestroy(event_start));
    CUDA_CHECK(cudaEventDestroy(event_stop));

    auto time_end_all = std::chrono::high_resolution_clock::now();
    float total_ms = std::chrono::duration<float, std::milli>(time_end_all - time_start_all).count();

    stats.backend = "cuda";
    stats.nx = grid.nx;
    stats.ny = grid.ny;
    stats.nz = grid.nz;
    stats.total_cells = total_cells;
    stats.active_cells = active_cell_count;
    stats.upload_ms = upload_ms;
    stats.marking_ms = marking_ms;
    stats.compaction_ms = compaction_ms;
    stats.qef_ms = qef_ms;
    stats.face_emission_ms = face_emission_ms;
    stats.download_ms = download_ms;
    stats.total_ms = total_ms;
    stats.vertex_count = active_cell_count;
    stats.face_count = mesh.faces.size() / 4;
    stats.qef_fallback_count = host_counters.qef_fallbacks;
    stats.clamp_count = host_counters.out_of_cell_clamps;

    return mesh;
}
