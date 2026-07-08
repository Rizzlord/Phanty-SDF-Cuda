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

#define CUDA_CHECK_SPARSE(val) { cuda_check_err_sparse((val), #val, __FILE__, __LINE__); }
inline void cuda_check_err_sparse(cudaError_t err, const char* const func, const char* const file, int const line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << err << "(" << cudaGetErrorString(err) << ") \"" << func << "\"\n";
        exit(EXIT_FAILURE);
    }
}

struct SparseGridParams {
    int nx, ny, nz;
    int nbx, nby, nbz;
    int brick_size;
    float vx, vy, vz;
    float ox, oy, oz;
    NormalComputationMode normal_mode;
    bool multi_vertex_cells;
};

struct SparseDeviceCounters {
    int qef_fallbacks;
    int out_of_cell_clamps;
    int invalid_count;
    int ambiguous_cells;
    int multi_vertex_cells;
    int one_vertex_cells;
    int two_vertex_cells;
    int split_rejections;
    int bad_qef_count;
    int faces_skipped;
};

__device__ __host__ inline int sparse_grid_index(int ix, int iy, int iz, int nx, int ny, int nz) {
    return ix + nx * (iy + ny * iz);
}

__device__ __host__ inline int sparse_cell_index(int ix, int iy, int iz, int nx, int ny, int nz) {
    return ix + (nx - 1) * (iy + (ny - 1) * iz);
}

__device__ __host__ inline int sparse_brick_index(int bx, int by, int bz, int nbx, int nby, int nbz) {
    return bx + nbx * (by + nby * bz);
}

__device__ __host__ inline void sparse_unflatten_brick(int b_idx, int nbx, int nby, int nbz, int& bx, int& by, int& bz) {
    bx = b_idx % nbx;
    by = (b_idx / nbx) % nby;
    bz = b_idx / (nbx * nby);
}

__device__ __host__ inline void sparse_unflatten_cell(int c_idx, int nx, int ny, int nz, int& ix, int& iy, int& iz) {
    int cx = nx - 1;
    int cy = ny - 1;
    ix = c_idx % cx;
    iy = (c_idx / cx) % cy;
    iz = c_idx / (cx * cy);
}

__device__ inline float3 sparse_operator_plus(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ inline float3 sparse_operator_minus(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ inline float3 sparse_operator_mul(float a, float3 b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}
__device__ inline float sparse_dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ inline float3 sparse_cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ inline float3 sparse_gradient_trilinear(
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

__device__ inline float3 sparse_get_gradient(const float* sdf, int gi, int gj, int gk, SparseGridParams params) {
    int im = (gi > 0) ? gi - 1 : 0;
    int ip = (gi < params.nx - 1) ? gi + 1 : params.nx - 1;
    int jm = (gj > 0) ? gj - 1 : 0;
    int jp = (gj < params.ny - 1) ? gj + 1 : params.ny - 1;
    int km = (gk > 0) ? gk - 1 : 0;
    int kp = (gk < params.nz - 1) ? gk + 1 : params.nz - 1;

    float s_im = sdf[sparse_grid_index(im, gj, gk, params.nx, params.ny, params.nz)];
    float s_ip = sdf[sparse_grid_index(ip, gj, gk, params.nx, params.ny, params.nz)];
    float s_jm = sdf[sparse_grid_index(gi, jm, gk, params.nx, params.ny, params.nz)];
    float s_jp = sdf[sparse_grid_index(gi, jp, gk, params.nx, params.ny, params.nz)];
    float s_km = sdf[sparse_grid_index(gi, gj, km, params.nx, params.ny, params.nz)];
    float s_kp = sdf[sparse_grid_index(gi, gj, kp, params.nx, params.ny, params.nz)];

    float3 g = make_float3(0.0f, 0.0f, 0.0f);
    float dx = (ip - im) * params.vx;
    float dy = (jp - jm) * params.vy;
    float dz = (kp - km) * params.vz;

    if (dx > 1e-12f) g.x = (s_ip - s_im) / dx;
    if (dy > 1e-12f) g.y = (s_jp - s_jm) / dy;
    if (dz > 1e-12f) g.z = (s_kp - s_km) / dz;
    return g;
}

__device__ void sparse_jacobi_solve_3x3(
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

__global__ void mark_active_bricks_kernel(
    const float* d_sdf,
    uint8_t* d_active_brick_flags,
    int total_bricks,
    SparseGridParams params
) {
    int b_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (b_idx >= total_bricks) return;

    int bx = 0, by = 0, bz = 0;
    sparse_unflatten_brick(b_idx, params.nbx, params.nby, params.nbz, bx, by, bz);

    int start_ix = bx * params.brick_size;
    int start_iy = by * params.brick_size;
    int start_iz = bz * params.brick_size;

    int end_ix = min(start_ix + params.brick_size + 1, params.nx);
    int end_iy = min(start_iy + params.brick_size + 1, params.ny);
    int end_iz = min(start_iz + params.brick_size + 1, params.nz);

    bool has_neg = false;
    bool has_pos = false;
    bool close_to_surface = false;
    float brick_diag = sqrtf(3.0f) * params.brick_size * fmaxf(params.vx, fmaxf(params.vy, params.vz));

    for (int iz = start_iz; iz < end_iz && (!has_neg || !has_pos || !close_to_surface); ++iz) {
        for (int iy = start_iy; iy < end_iy && (!has_neg || !has_pos || !close_to_surface); ++iy) {
            for (int ix = start_ix; ix < end_ix && (!has_neg || !has_pos || !close_to_surface); ++ix) {
                float val = d_sdf[sparse_grid_index(ix, iy, iz, params.nx, params.ny, params.nz)];
                if (val < 0.0f) has_neg = true;
                if (val > 0.0f) has_pos = true;
                if (fabsf(val) <= 2.0f * brick_diag) close_to_surface = true;
            }
        }
    }

    d_active_brick_flags[b_idx] = (has_neg && has_pos) ? 1 : 0;
}

__device__ inline bool sparse_is_thin_edge(
    float fa, float fb, int ax, int ay, int az, int bx, int by, int bz,
    const float* d_sdf, SparseGridParams params
) {
    if (fa * fb <= 0.0f) return false;
    if (fabsf(fa) > 0.45f * params.vx || fabsf(fb) > 0.45f * params.vx) return false;
    float3 ga = sparse_get_gradient(d_sdf, ax, ay, az, params);
    float3 gb = sparse_get_gradient(d_sdf, bx, by, bz, params);
    return (sparse_dot(ga, gb) < -0.3f);
}

__global__ void mark_active_cells_sparse_kernel(
    const int* d_active_brick_indices,
    const float* d_sdf,
    uint8_t* d_active_cell_flags,
    int active_brick_count,
    SparseGridParams params
) {
    int active_brick_idx = blockIdx.x;
    if (active_brick_idx >= active_brick_count) return;

    int b_idx = d_active_brick_indices[active_brick_idx];
    int bx = 0, by = 0, bz = 0;
    sparse_unflatten_brick(b_idx, params.nbx, params.nby, params.nbz, bx, by, bz);

    int start_ix = bx * params.brick_size;
    int start_iy = by * params.brick_size;
    int start_iz = bz * params.brick_size;

    int cells_per_brick = params.brick_size * params.brick_size * params.brick_size;
    int tid = threadIdx.x;

    const int corner_offsets[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
    };

    for (int c = tid; c < cells_per_brick; c += blockDim.x) {
        int c_rel_x = c % params.brick_size;
        int c_rel_y = (c / params.brick_size) % params.brick_size;
        int c_rel_z = c / (params.brick_size * params.brick_size);

        int ix = start_ix + c_rel_x;
        int iy = start_iy + c_rel_y;
        int iz = start_iz + c_rel_z;

        if (ix < params.nx - 1 && iy < params.ny - 1 && iz < params.nz - 1) {
            bool has_neg = false;
            bool has_pos = false;
            bool very_close = false;

            for (int k = 0; k < 8; ++k) {
                int g_ix = ix + corner_offsets[k][0];
                int g_iy = iy + corner_offsets[k][1];
                int g_iz = iz + corner_offsets[k][2];
                float val = d_sdf[sparse_grid_index(g_ix, g_iy, g_iz, params.nx, params.ny, params.nz)];
                if (val < 0.0f) has_neg = true;
                if (val > 0.0f) has_pos = true;
                if (fabsf(val) <= 0.45f * params.vx) very_close = true;
            }

            bool is_active = (has_neg && has_pos) || very_close;

            if (is_active) {
                d_active_cell_flags[sparse_cell_index(ix, iy, iz, params.nx, params.ny, params.nz)] = 1;
            }
        }
    }
}

__device__ inline float3 sparse_solve_qef_cluster(
    const float3* pts,
    const float3* normals,
    int m,
    float3 cell_min,
    float3 cell_max,
    SparseGridParams params,
    SparseDeviceCounters* d_counters
) {
    if (m == 0) {
        return sparse_operator_plus(cell_min, sparse_operator_mul(0.5f, make_float3(params.vx, params.vy, params.vz)));
    }
    float3 centroid = make_float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < m; ++i) {
        centroid = sparse_operator_plus(centroid, pts[i]);
    }
    centroid = sparse_operator_mul(1.0f / (float)m, centroid);

    float B[9] = {0.0f};
    float d[3] = {0.0f};
    for (int i = 0; i < m; ++i) {
        float3 n = normals[i];
        float3 diff = sparse_operator_minus(pts[i], centroid);
        float dot_val = sparse_dot(n, diff);

        B[0] += n.x * n.x; B[1] += n.x * n.y; B[2] += n.x * n.z;
        B[3] += n.y * n.x; B[4] += n.y * n.y; B[5] += n.y * n.z;
        B[6] += n.z * n.x; B[7] += n.z * n.y; B[8] += n.z * n.z;

        d[0] += n.x * dot_val;
        d[1] += n.y * dot_val;
        d[2] += n.z * dot_val;
    }

    float y[3] = {0.0f};
    bool singular = false;
    sparse_jacobi_solve_3x3(B, d, 0.01f, y, singular);

    float3 vertex = sparse_operator_plus(centroid, make_float3(y[0], y[1], y[2]));

    bool has_nan_inf = isnan(vertex.x) || isnan(vertex.y) || isnan(vertex.z) ||
                       isinf(vertex.x) || isinf(vertex.y) || isinf(vertex.z);

    if (singular || has_nan_inf) {
        atomicAdd(&d_counters->qef_fallbacks, 1);
        vertex = centroid;
    }

    if (isnan(vertex.x) || isnan(vertex.y) || isnan(vertex.z) ||
        isinf(vertex.x) || isinf(vertex.y) || isinf(vertex.z)) {
        atomicAdd(&d_counters->invalid_count, 1);
        vertex = sparse_operator_plus(cell_min, sparse_operator_mul(0.5f, make_float3(params.vx, params.vy, params.vz)));
    }

    if (vertex.x < cell_min.x || vertex.x > cell_max.x ||
        vertex.y < cell_min.y || vertex.y > cell_max.y ||
        vertex.z < cell_min.z || vertex.z > cell_max.z) {
        atomicAdd(&d_counters->out_of_cell_clamps, 1);
        vertex.x = fmaxf(cell_min.x, fminf(cell_max.x, vertex.x));
        vertex.y = fmaxf(cell_min.y, fminf(cell_max.y, vertex.y));
        vertex.z = fmaxf(cell_min.z, fminf(cell_max.z, vertex.z));
    }
    return vertex;
}

__global__ void count_cell_vertices_sparse_kernel(
    const int* d_active_cell_indices,
    const float* d_sdf,
    const float3* d_precomputed_gradients,
    int* d_active_cell_vertex_counts,
    int active_cell_count,
    SparseGridParams params,
    SparseDeviceCounters* d_counters
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= active_cell_count) return;

    if (!params.multi_vertex_cells) {
        d_active_cell_vertex_counts[idx] = 1;
        return;
    }

    int cell_idx = d_active_cell_indices[idx];
    int ix = 0, iy = 0, iz = 0;
    sparse_unflatten_cell(cell_idx, params.nx, params.ny, params.nz, ix, iy, iz);

    const int corner_offsets[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
    };

    const int edge_pairs[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7}
    };

    float corner_sdf_vals[8];
    float3 corner_pos_vals[8];
    for (int c = 0; c < 8; ++c) {
        int c_ix = ix + corner_offsets[c][0];
        int c_iy = iy + corner_offsets[c][1];
        int c_iz = iz + corner_offsets[c][2];
        corner_sdf_vals[c] = d_sdf[sparse_grid_index(c_ix, c_iy, c_iz, params.nx, params.ny, params.nz)];
        corner_pos_vals[c] = make_float3(params.ox + c_ix * params.vx, params.oy + c_iy * params.vy, params.oz + c_iz * params.vz);
    }

    float3 pts[12];
    float3 normals[12];
    float3 centroid = make_float3(0.0f, 0.0f, 0.0f);
    int m = 0;

    for (int e = 0; e < 12; ++e) {
        int a = edge_pairs[e][0];
        int b = edge_pairs[e][1];
        float fa = corner_sdf_vals[a];
        float fb = corner_sdf_vals[b];
        int a_ix = ix + corner_offsets[a][0];
        int a_iy = iy + corner_offsets[a][1];
        int a_iz = iz + corner_offsets[a][2];
        int b_ix = ix + corner_offsets[b][0];
        int b_iy = iy + corner_offsets[b][1];
        int b_iz = iz + corner_offsets[b][2];

        bool is_crossing = (fa * fb <= 0.0f);
        bool is_thin = (!is_crossing && sparse_is_thin_edge(fa, fb, a_ix, a_iy, a_iz, b_ix, b_iy, b_iz, d_sdf, params));

        if (is_crossing || is_thin) {
            float t = 0.5f;
            if (is_crossing) {
                t = fa / (fa - fb + 1e-12f);
            } else {
                t = fabsf(fa) / (fabsf(fa) + fabsf(fb) + 1e-12f);
            }
            t = fmaxf(0.0f, fminf(1.0f, t));
            float3 pa = corner_pos_vals[a];
            float3 pb = corner_pos_vals[b];
            float3 p = sparse_operator_plus(sparse_operator_mul(1.0f - t, pa), sparse_operator_mul(t, pb));

            if (is_thin) {
                float3 ga = sparse_get_gradient(d_sdf, a_ix, a_iy, a_iz, params);
                float3 gb = sparse_get_gradient(d_sdf, b_ix, b_iy, b_iz, params);
                float ga_len = sqrtf(sparse_dot(ga, ga));
                float gb_len = sqrtf(sparse_dot(gb, gb));
                if (ga_len > 1e-6f) ga = sparse_operator_mul(1.0f / ga_len, ga);
                if (gb_len > 1e-6f) gb = sparse_operator_mul(1.0f / gb_len, gb);
                if (m < 11) {
                    pts[m] = p; normals[m] = ga; centroid = sparse_operator_plus(centroid, p); m++;
                    pts[m] = p; normals[m] = gb; centroid = sparse_operator_plus(centroid, p); m++;
                }
            } else {
                float3 n = make_float3(0.0f, 0.0f, 0.0f);
                if (params.normal_mode == NormalComputationMode::FiniteDifference) {
                    n = sparse_gradient_trilinear(corner_sdf_vals, corner_pos_vals, p);
                } else if (params.normal_mode == NormalComputationMode::EdgeGradient) {
                    float3 ga = sparse_get_gradient(d_sdf, a_ix, a_iy, a_iz, params);
                    float3 gb = sparse_get_gradient(d_sdf, b_ix, b_iy, b_iz, params);
                    n = sparse_operator_plus(sparse_operator_mul(1.0f - t, ga), sparse_operator_mul(t, gb));
                } else if (params.normal_mode == NormalComputationMode::PrecomputedGradient && d_precomputed_gradients != nullptr) {
                    float3 corner_grad_vals[8];
                    for (int c = 0; c < 8; ++c) {
                        int c_ix = ix + corner_offsets[c][0];
                        int c_iy = iy + corner_offsets[c][1];
                        int c_iz = iz + corner_offsets[c][2];
                        corner_grad_vals[c] = d_precomputed_gradients[sparse_grid_index(c_ix, c_iy, c_iz, params.nx, params.ny, params.nz)];
                    }
                    float u = (p.x - corner_pos_vals[0].x) / params.vx;
                    float v = (p.y - corner_pos_vals[0].y) / params.vy;
                    float w = (p.z - corner_pos_vals[0].z) / params.vz;
                    u = fmaxf(0.0f, fminf(1.0f, u));
                    v = fmaxf(0.0f, fminf(1.0f, v));
                    w = fmaxf(0.0f, fminf(1.0f, w));
                    float3 g00 = sparse_operator_plus(sparse_operator_mul(1.0f - u, corner_grad_vals[0]), sparse_operator_mul(u, corner_grad_vals[1]));
                    float3 g10 = sparse_operator_plus(sparse_operator_mul(1.0f - u, corner_grad_vals[3]), sparse_operator_mul(u, corner_grad_vals[2]));
                    float3 g01 = sparse_operator_plus(sparse_operator_mul(1.0f - u, corner_grad_vals[4]), sparse_operator_mul(u, corner_grad_vals[5]));
                    float3 g11 = sparse_operator_plus(sparse_operator_mul(1.0f - u, corner_grad_vals[7]), sparse_operator_mul(u, corner_grad_vals[6]));
                    float3 g0 = sparse_operator_plus(sparse_operator_mul(1.0f - v, g00), sparse_operator_mul(v, g10));
                    float3 g1 = sparse_operator_plus(sparse_operator_mul(1.0f - v, g01), sparse_operator_mul(v, g11));
                    n = sparse_operator_plus(sparse_operator_mul(1.0f - w, g0), sparse_operator_mul(w, g1));
                } else {
                    n = sparse_gradient_trilinear(corner_sdf_vals, corner_pos_vals, p);
                }
                float n_len = sqrtf(sparse_dot(n, n));
                if (n_len > 1e-6f) n = sparse_operator_mul(1.0f / n_len, n);
                if (m < 12) {
                    pts[m] = p; normals[m] = n; centroid = sparse_operator_plus(centroid, p); m++;
                }
            }
        }
    }

    if (m < 2) {
        d_active_cell_vertex_counts[idx] = 1;
        atomicAdd(&d_counters->one_vertex_cells, 1);
        return;
    }

    int best_A = -1;
    int best_B = -1;
    float min_dot = 1.0f;
    for (int i = 0; i < m; ++i) {
        for (int j = i + 1; j < m; ++j) {
            float dt = sparse_dot(normals[i], normals[j]);
            if (dt < min_dot) {
                min_dot = dt;
                best_A = i;
                best_B = j;
            }
        }
    }

    if (best_A < 0 || best_B < 0 || min_dot >= -0.3f) {
        d_active_cell_vertex_counts[idx] = 1;
        atomicAdd(&d_counters->one_vertex_cells, 1);
        return;
    }

    float3 nA = normals[best_A];
    float3 nB = normals[best_B];
    int countA = 0;
    int countB = 0;
    for (int i = 0; i < m; ++i) {
        if (sparse_dot(normals[i], nA) >= sparse_dot(normals[i], nB)) {
            countA++;
        } else {
            countB++;
        }
    }

    if (countA < 2 || countB < 2) {
        d_active_cell_vertex_counts[idx] = 1;
        atomicAdd(&d_counters->split_rejections, 1);
        atomicAdd(&d_counters->one_vertex_cells, 1);
        return;
    }

    d_active_cell_vertex_counts[idx] = 2;
    atomicAdd(&d_counters->ambiguous_cells, 1);
    atomicAdd(&d_counters->multi_vertex_cells, 1);
    atomicAdd(&d_counters->two_vertex_cells, 1);
}

__global__ void compute_vertices_sparse_kernel(
    const int* d_active_cell_indices,
    const float* d_sdf,
    const float3* d_precomputed_gradients,
    float3* d_vertices,
    float3* d_vertex_normals,
    int* d_cell_to_vertex_map,
    int* d_cell_vertex_counts_map,
    const int* d_scanned_vertex_counts,
    const int* d_active_cell_vertex_counts,
    int active_cell_count,
    SparseGridParams params,
    SparseDeviceCounters* d_counters
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= active_cell_count) return;

    int cell_idx = d_active_cell_indices[idx];
    int start_v = (d_scanned_vertex_counts != nullptr && params.multi_vertex_cells) ? d_scanned_vertex_counts[idx] : idx;
    int count_v = (d_active_cell_vertex_counts != nullptr && params.multi_vertex_cells) ? d_active_cell_vertex_counts[idx] : 1;

    d_cell_to_vertex_map[cell_idx] = start_v;
    if (d_cell_vertex_counts_map != nullptr) {
        d_cell_vertex_counts_map[cell_idx] = count_v;
    }

    int ix = 0, iy = 0, iz = 0;
    sparse_unflatten_cell(cell_idx, params.nx, params.ny, params.nz, ix, iy, iz);

    const int corner_offsets[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
    };

    const int edge_pairs[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7}
    };

    float corner_sdf_vals[8];
    float3 corner_pos_vals[8];
    for (int c = 0; c < 8; ++c) {
        int c_ix = ix + corner_offsets[c][0];
        int c_iy = iy + corner_offsets[c][1];
        int c_iz = iz + corner_offsets[c][2];
        corner_sdf_vals[c] = d_sdf[sparse_grid_index(c_ix, c_iy, c_iz, params.nx, params.ny, params.nz)];
        corner_pos_vals[c] = make_float3(params.ox + c_ix * params.vx, params.oy + c_iy * params.vy, params.oz + c_iz * params.vz);
    }

    float3 pts[12];
    float3 normals[12];
    float3 centroid = make_float3(0.0f, 0.0f, 0.0f);
    int m = 0;

    for (int e = 0; e < 12; ++e) {
        int a = edge_pairs[e][0];
        int b = edge_pairs[e][1];
        float fa = corner_sdf_vals[a];
        float fb = corner_sdf_vals[b];
        int a_ix = ix + corner_offsets[a][0];
        int a_iy = iy + corner_offsets[a][1];
        int a_iz = iz + corner_offsets[a][2];
        int b_ix = ix + corner_offsets[b][0];
        int b_iy = iy + corner_offsets[b][1];
        int b_iz = iz + corner_offsets[b][2];

        bool is_crossing = (fa * fb <= 0.0f);
        bool is_thin = (!is_crossing && sparse_is_thin_edge(fa, fb, a_ix, a_iy, a_iz, b_ix, b_iy, b_iz, d_sdf, params));

        if (is_crossing || is_thin) {
            float t = 0.5f;
            if (is_crossing) {
                t = fa / (fa - fb + 1e-12f);
            } else {
                t = fabsf(fa) / (fabsf(fa) + fabsf(fb) + 1e-12f);
            }
            t = fmaxf(0.0f, fminf(1.0f, t));
            float3 pa = corner_pos_vals[a];
            float3 pb = corner_pos_vals[b];
            float3 p = sparse_operator_plus(sparse_operator_mul(1.0f - t, pa), sparse_operator_mul(t, pb));

            if (is_thin) {
                float3 ga = sparse_get_gradient(d_sdf, a_ix, a_iy, a_iz, params);
                float3 gb = sparse_get_gradient(d_sdf, b_ix, b_iy, b_iz, params);
                float ga_len = sqrtf(sparse_dot(ga, ga));
                float gb_len = sqrtf(sparse_dot(gb, gb));
                if (ga_len > 1e-6f) ga = sparse_operator_mul(1.0f / ga_len, ga);
                if (gb_len > 1e-6f) gb = sparse_operator_mul(1.0f / gb_len, gb);
                if (m < 11) {
                    pts[m] = p; normals[m] = ga; centroid = sparse_operator_plus(centroid, p); m++;
                    pts[m] = p; normals[m] = gb; centroid = sparse_operator_plus(centroid, p); m++;
                }
            } else {
                float3 n = make_float3(0.0f, 0.0f, 0.0f);
                if (params.normal_mode == NormalComputationMode::FiniteDifference) {
                    n = sparse_gradient_trilinear(corner_sdf_vals, corner_pos_vals, p);
                } else if (params.normal_mode == NormalComputationMode::EdgeGradient) {
                    float3 ga = sparse_get_gradient(d_sdf, a_ix, a_iy, a_iz, params);
                    float3 gb = sparse_get_gradient(d_sdf, b_ix, b_iy, b_iz, params);
                    n = sparse_operator_plus(sparse_operator_mul(1.0f - t, ga), sparse_operator_mul(t, gb));
                } else if (params.normal_mode == NormalComputationMode::PrecomputedGradient && d_precomputed_gradients != nullptr) {
                    float3 corner_grad_vals[8];
                    for (int c = 0; c < 8; ++c) {
                        int c_ix = ix + corner_offsets[c][0];
                        int c_iy = iy + corner_offsets[c][1];
                        int c_iz = iz + corner_offsets[c][2];
                        corner_grad_vals[c] = d_precomputed_gradients[sparse_grid_index(c_ix, c_iy, c_iz, params.nx, params.ny, params.nz)];
                    }
                    float u = (p.x - corner_pos_vals[0].x) / params.vx;
                    float v = (p.y - corner_pos_vals[0].y) / params.vy;
                    float w = (p.z - corner_pos_vals[0].z) / params.vz;
                    u = fmaxf(0.0f, fminf(1.0f, u));
                    v = fmaxf(0.0f, fminf(1.0f, v));
                    w = fmaxf(0.0f, fminf(1.0f, w));
                    float3 g00 = sparse_operator_plus(sparse_operator_mul(1.0f - u, corner_grad_vals[0]), sparse_operator_mul(u, corner_grad_vals[1]));
                    float3 g10 = sparse_operator_plus(sparse_operator_mul(1.0f - u, corner_grad_vals[3]), sparse_operator_mul(u, corner_grad_vals[2]));
                    float3 g01 = sparse_operator_plus(sparse_operator_mul(1.0f - u, corner_grad_vals[4]), sparse_operator_mul(u, corner_grad_vals[5]));
                    float3 g11 = sparse_operator_plus(sparse_operator_mul(1.0f - u, corner_grad_vals[7]), sparse_operator_mul(u, corner_grad_vals[6]));
                    float3 g0 = sparse_operator_plus(sparse_operator_mul(1.0f - v, g00), sparse_operator_mul(v, g10));
                    float3 g1 = sparse_operator_plus(sparse_operator_mul(1.0f - v, g01), sparse_operator_mul(v, g11));
                    n = sparse_operator_plus(sparse_operator_mul(1.0f - w, g0), sparse_operator_mul(w, g1));
                } else {
                    n = sparse_gradient_trilinear(corner_sdf_vals, corner_pos_vals, p);
                }
                float n_len = sqrtf(sparse_dot(n, n));
                if (n_len > 1e-6f) n = sparse_operator_mul(1.0f / n_len, n);
                if (m < 12) {
                    pts[m] = p; normals[m] = n; centroid = sparse_operator_plus(centroid, p); m++;
                }
            }
        }
    }

    float3 cell_min = corner_pos_vals[0];
    float3 cell_max = corner_pos_vals[6];

    if (count_v == 1 || m < 2) {
        float3 v0 = sparse_solve_qef_cluster(pts, normals, m, cell_min, cell_max, params, d_counters);
        d_vertices[start_v] = v0;
        if (d_vertex_normals != nullptr) {
            float3 avg_n = make_float3(0.0f, 0.0f, 0.0f);
            for (int i = 0; i < m; ++i) avg_n = sparse_operator_plus(avg_n, normals[i]);
            float len = sqrtf(sparse_dot(avg_n, avg_n));
            d_vertex_normals[start_v] = (len > 1e-6f) ? sparse_operator_mul(1.0f / len, avg_n) : make_float3(0.0f, 1.0f, 0.0f);
        }
        return;
    }

    int best_A = -1;
    int best_B = -1;
    float min_dot = 1.0f;
    for (int i = 0; i < m; ++i) {
        for (int j = i + 1; j < m; ++j) {
            float dt = sparse_dot(normals[i], normals[j]);
            if (dt < min_dot) {
                min_dot = dt;
                best_A = i;
                best_B = j;
            }
        }
    }

    if (best_A < 0 || best_B < 0) {
        float3 v0 = sparse_solve_qef_cluster(pts, normals, m, cell_min, cell_max, params, d_counters);
        d_vertices[start_v] = v0;
        if (d_vertex_normals != nullptr) d_vertex_normals[start_v] = make_float3(0.0f, 1.0f, 0.0f);
        return;
    }

    float3 nA = normals[best_A];
    float3 nB = normals[best_B];

    float3 pts0[12];
    float3 normals0[12];
    int m0 = 0;
    float3 pts1[12];
    float3 normals1[12];
    int m1 = 0;

    for (int i = 0; i < m; ++i) {
        if (sparse_dot(normals[i], nA) >= sparse_dot(normals[i], nB)) {
            pts0[m0] = pts[i]; normals0[m0] = normals[i]; m0++;
        } else {
            pts1[m1] = pts[i]; normals1[m1] = normals[i]; m1++;
        }
    }

    float3 v0 = sparse_solve_qef_cluster(pts0, normals0, m0, cell_min, cell_max, params, d_counters);
    float3 v1 = sparse_solve_qef_cluster(pts1, normals1, m1, cell_min, cell_max, params, d_counters);

    d_vertices[start_v + 0] = v0;
    d_vertices[start_v + 1] = v1;

    if (d_vertex_normals != nullptr) {
        float3 avg0 = make_float3(0.0f, 0.0f, 0.0f);
        for (int i = 0; i < m0; ++i) avg0 = sparse_operator_plus(avg0, normals0[i]);
        float len0 = sqrtf(sparse_dot(avg0, avg0));
        d_vertex_normals[start_v + 0] = (len0 > 1e-6f) ? sparse_operator_mul(1.0f / len0, avg0) : nA;

        float3 avg1 = make_float3(0.0f, 0.0f, 0.0f);
        for (int i = 0; i < m1; ++i) avg1 = sparse_operator_plus(avg1, normals1[i]);
        float len1 = sqrtf(sparse_dot(avg1, avg1));
        d_vertex_normals[start_v + 1] = (len1 > 1e-6f) ? sparse_operator_mul(1.0f / len1, avg1) : nB;
    }
}

__device__ inline int sparse_select_cell_vertex(
    int cell_idx,
    const int* d_cell_to_vertex_map,
    const int* d_cell_vertex_counts_map,
    const float3* d_vertex_normals,
    float3 g_n,
    bool multi_vertex_cells
) {
    int start = d_cell_to_vertex_map[cell_idx];
    if (start < 0) return -1;
    if (!multi_vertex_cells || d_cell_vertex_counts_map == nullptr || d_vertex_normals == nullptr) return start;
    int count = d_cell_vertex_counts_map[cell_idx];
    if (count <= 0) return -1;
    if (count == 1) return start;
    float dot0 = sparse_dot(g_n, d_vertex_normals[start + 0]);
    float dot1 = sparse_dot(g_n, d_vertex_normals[start + 1]);
    int best = (dot0 >= dot1) ? (start + 0) : (start + 1);
    float best_dot = fmaxf(dot0, dot1);
    if (best_dot < -0.5f) {
        return -1;
    }
    return best;
}

__global__ void count_sparse_faces_kernel(
    const int* d_active_cell_indices,
    const float* d_sdf,
    const int* d_cell_to_vertex_map,
    const int* d_cell_vertex_counts_map,
    const float3* d_vertex_normals,
    int* d_active_cell_face_counts,
    int active_cell_count,
    SparseGridParams params
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= active_cell_count) return;

    int cell_idx = d_active_cell_indices[idx];
    int ix = 0, iy = 0, iz = 0;
    sparse_unflatten_cell(cell_idx, params.nx, params.ny, params.nz, ix, iy, iz);

    int count = 0;

    if (iy >= 1 && iz >= 1 && ix < params.nx - 1) {
        int g0 = sparse_grid_index(ix, iy, iz, params.nx, params.ny, params.nz);
        int g1 = sparse_grid_index(ix + 1, iy, iz, params.nx, params.ny, params.nz);
        bool is_crossing = (d_sdf[g0] * d_sdf[g1] < 0.0f);
        bool is_thin = (!is_crossing && sparse_is_thin_edge(d_sdf[g0], d_sdf[g1], ix, iy, iz, ix + 1, iy, iz, d_sdf, params));
        if (is_crossing || is_thin) {
            float3 g_n = sparse_operator_plus(sparse_get_gradient(d_sdf, ix, iy, iz, params), sparse_get_gradient(d_sdf, ix + 1, iy, iz, params));
            int c0 = sparse_cell_index(ix, iy - 1, iz - 1, params.nx, params.ny, params.nz);
            int c1 = sparse_cell_index(ix, iy, iz - 1, params.nx, params.ny, params.nz);
            int c2 = sparse_cell_index(ix, iy, iz, params.nx, params.ny, params.nz);
            int c3 = sparse_cell_index(ix, iy - 1, iz, params.nx, params.ny, params.nz);
            int v0 = sparse_select_cell_vertex(c0, d_cell_to_vertex_map, d_cell_vertex_counts_map, d_vertex_normals, g_n, params.multi_vertex_cells);
            int v1 = sparse_select_cell_vertex(c1, d_cell_to_vertex_map, d_cell_vertex_counts_map, d_vertex_normals, g_n, params.multi_vertex_cells);
            int v2 = sparse_select_cell_vertex(c2, d_cell_to_vertex_map, d_cell_vertex_counts_map, d_vertex_normals, g_n, params.multi_vertex_cells);
            int v3 = sparse_select_cell_vertex(c3, d_cell_to_vertex_map, d_cell_vertex_counts_map, d_vertex_normals, g_n, params.multi_vertex_cells);
            if (v0 >= 0 && v1 >= 0 && v2 >= 0 && v3 >= 0) {
                count += is_thin ? 2 : 1;
            }
        }
    }

    if (ix >= 1 && iz >= 1 && iy < params.ny - 1) {
        int g0 = sparse_grid_index(ix, iy, iz, params.nx, params.ny, params.nz);
        int g1 = sparse_grid_index(ix, iy + 1, iz, params.nx, params.ny, params.nz);
        bool is_crossing = (d_sdf[g0] * d_sdf[g1] < 0.0f);
        bool is_thin = (!is_crossing && sparse_is_thin_edge(d_sdf[g0], d_sdf[g1], ix, iy, iz, ix, iy + 1, iz, d_sdf, params));
        if (is_crossing || is_thin) {
            float3 g_n = sparse_operator_plus(sparse_get_gradient(d_sdf, ix, iy, iz, params), sparse_get_gradient(d_sdf, ix, iy + 1, iz, params));
            int c0 = sparse_cell_index(ix - 1, iy, iz - 1, params.nx, params.ny, params.nz);
            int c1 = sparse_cell_index(ix, iy, iz - 1, params.nx, params.ny, params.nz);
            int c2 = sparse_cell_index(ix, iy, iz, params.nx, params.ny, params.nz);
            int c3 = sparse_cell_index(ix - 1, iy, iz, params.nx, params.ny, params.nz);
            int v0 = sparse_select_cell_vertex(c0, d_cell_to_vertex_map, d_cell_vertex_counts_map, d_vertex_normals, g_n, params.multi_vertex_cells);
            int v1 = sparse_select_cell_vertex(c1, d_cell_to_vertex_map, d_cell_vertex_counts_map, d_vertex_normals, g_n, params.multi_vertex_cells);
            int v2 = sparse_select_cell_vertex(c2, d_cell_to_vertex_map, d_cell_vertex_counts_map, d_vertex_normals, g_n, params.multi_vertex_cells);
            int v3 = sparse_select_cell_vertex(c3, d_cell_to_vertex_map, d_cell_vertex_counts_map, d_vertex_normals, g_n, params.multi_vertex_cells);
            if (v0 >= 0 && v1 >= 0 && v2 >= 0 && v3 >= 0) {
                count += is_thin ? 2 : 1;
            }
        }
    }

    if (ix >= 1 && iy >= 1 && iz < params.nz - 1) {
        int g0 = sparse_grid_index(ix, iy, iz, params.nx, params.ny, params.nz);
        int g1 = sparse_grid_index(ix, iy, iz + 1, params.nx, params.ny, params.nz);
        bool is_crossing = (d_sdf[g0] * d_sdf[g1] < 0.0f);
        bool is_thin = (!is_crossing && sparse_is_thin_edge(d_sdf[g0], d_sdf[g1], ix, iy, iz, ix, iy, iz + 1, d_sdf, params));
        if (is_crossing || is_thin) {
            float3 g_n = sparse_operator_plus(sparse_get_gradient(d_sdf, ix, iy, iz, params), sparse_get_gradient(d_sdf, ix, iy, iz + 1, params));
            int c0 = sparse_cell_index(ix - 1, iy - 1, iz, params.nx, params.ny, params.nz);
            int c1 = sparse_cell_index(ix, iy - 1, iz, params.nx, params.ny, params.nz);
            int c2 = sparse_cell_index(ix, iy, iz, params.nx, params.ny, params.nz);
            int c3 = sparse_cell_index(ix - 1, iy, iz, params.nx, params.ny, params.nz);
            int v0 = sparse_select_cell_vertex(c0, d_cell_to_vertex_map, d_cell_vertex_counts_map, d_vertex_normals, g_n, params.multi_vertex_cells);
            int v1 = sparse_select_cell_vertex(c1, d_cell_to_vertex_map, d_cell_vertex_counts_map, d_vertex_normals, g_n, params.multi_vertex_cells);
            int v2 = sparse_select_cell_vertex(c2, d_cell_to_vertex_map, d_cell_vertex_counts_map, d_vertex_normals, g_n, params.multi_vertex_cells);
            int v3 = sparse_select_cell_vertex(c3, d_cell_to_vertex_map, d_cell_vertex_counts_map, d_vertex_normals, g_n, params.multi_vertex_cells);
            if (v0 >= 0 && v1 >= 0 && v2 >= 0 && v3 >= 0) {
                count += is_thin ? 2 : 1;
            }
        }
    }

    d_active_cell_face_counts[idx] = count;
}

__global__ void emit_sparse_faces_kernel(
    const int* d_active_cell_indices,
    const float* d_sdf,
    const int* d_cell_to_vertex_map,
    const int* d_cell_vertex_counts_map,
    const float3* d_vertices,
    const float3* d_vertex_normals,
    const int* d_active_cell_face_counts,
    const int* d_scanned_face_counts,
    int* d_faces,
    int active_cell_count,
    SparseGridParams params,
    SparseDeviceCounters* d_counters
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= active_cell_count) return;

    if (d_active_cell_face_counts[idx] == 0) return;
    int out_idx = d_scanned_face_counts[idx];

    int cell_idx = d_active_cell_indices[idx];
    int ix = 0, iy = 0, iz = 0;
    sparse_unflatten_cell(cell_idx, params.nx, params.ny, params.nz, ix, iy, iz);

    if (iy >= 1 && iz >= 1 && ix < params.nx - 1) {
        int g0 = sparse_grid_index(ix, iy, iz, params.nx, params.ny, params.nz);
        int g1 = sparse_grid_index(ix + 1, iy, iz, params.nx, params.ny, params.nz);
        bool is_crossing = (d_sdf[g0] * d_sdf[g1] < 0.0f);
        bool is_thin = (!is_crossing && sparse_is_thin_edge(d_sdf[g0], d_sdf[g1], ix, iy, iz, ix + 1, iy, iz, d_sdf, params));
        if (is_crossing || is_thin) {
            float3 g_n = sparse_operator_plus(sparse_get_gradient(d_sdf, ix, iy, iz, params), sparse_get_gradient(d_sdf, ix + 1, iy, iz, params));
            int c0 = sparse_cell_index(ix, iy - 1, iz - 1, params.nx, params.ny, params.nz);
            int c1 = sparse_cell_index(ix, iy, iz - 1, params.nx, params.ny, params.nz);
            int c2 = sparse_cell_index(ix, iy, iz, params.nx, params.ny, params.nz);
            int c3 = sparse_cell_index(ix, iy - 1, iz, params.nx, params.ny, params.nz);
            int v0 = sparse_select_cell_vertex(c0, d_cell_to_vertex_map, d_cell_vertex_counts_map, d_vertex_normals, g_n, params.multi_vertex_cells);
            int v1 = sparse_select_cell_vertex(c1, d_cell_to_vertex_map, d_cell_vertex_counts_map, d_vertex_normals, g_n, params.multi_vertex_cells);
            int v2 = sparse_select_cell_vertex(c2, d_cell_to_vertex_map, d_cell_vertex_counts_map, d_vertex_normals, g_n, params.multi_vertex_cells);
            int v3 = sparse_select_cell_vertex(c3, d_cell_to_vertex_map, d_cell_vertex_counts_map, d_vertex_normals, g_n, params.multi_vertex_cells);
            if (v0 >= 0 && v1 >= 0 && v2 >= 0 && v3 >= 0) {
                float3 pa = d_vertices[v0];
                float3 pb = d_vertices[v1];
                float3 pc = d_vertices[v2];
                float3 pd = d_vertices[v3];
                float3 quad_n = sparse_cross(sparse_operator_minus(pb, pa), sparse_operator_minus(pd, pa));
                if (sparse_dot(quad_n, quad_n) < 1e-12f) {
                    quad_n = sparse_cross(sparse_operator_minus(pc, pa), sparse_operator_minus(pd, pa));
                }
                if (sparse_dot(g_n, g_n) > 0.0f && sparse_dot(quad_n, quad_n) > 0.0f) {
                    if (sparse_dot(quad_n, g_n) < 0.0f) {
                        int tmp = v1; v1 = v3; v3 = tmp;
                    }
                }
                d_faces[4 * out_idx + 0] = v0;
                d_faces[4 * out_idx + 1] = v1;
                d_faces[4 * out_idx + 2] = v2;
                d_faces[4 * out_idx + 3] = v3;
                out_idx++;
                if (is_thin) {
                    d_faces[4 * out_idx + 0] = v0;
                    d_faces[4 * out_idx + 1] = v3;
                    d_faces[4 * out_idx + 2] = v2;
                    d_faces[4 * out_idx + 3] = v1;
                    out_idx++;
                }
            } else if (params.multi_vertex_cells) {
                atomicAdd(&d_counters->faces_skipped, 1);
            }
        }
    }

    if (ix >= 1 && iz >= 1 && iy < params.ny - 1) {
        int g0 = sparse_grid_index(ix, iy, iz, params.nx, params.ny, params.nz);
        int g1 = sparse_grid_index(ix, iy + 1, iz, params.nx, params.ny, params.nz);
        bool is_crossing = (d_sdf[g0] * d_sdf[g1] < 0.0f);
        bool is_thin = (!is_crossing && sparse_is_thin_edge(d_sdf[g0], d_sdf[g1], ix, iy, iz, ix, iy + 1, iz, d_sdf, params));
        if (is_crossing || is_thin) {
            float3 g_n = sparse_operator_plus(sparse_get_gradient(d_sdf, ix, iy, iz, params), sparse_get_gradient(d_sdf, ix, iy + 1, iz, params));
            int c0 = sparse_cell_index(ix - 1, iy, iz - 1, params.nx, params.ny, params.nz);
            int c1 = sparse_cell_index(ix, iy, iz - 1, params.nx, params.ny, params.nz);
            int c2 = sparse_cell_index(ix, iy, iz, params.nx, params.ny, params.nz);
            int c3 = sparse_cell_index(ix - 1, iy, iz, params.nx, params.ny, params.nz);
            int v0 = sparse_select_cell_vertex(c0, d_cell_to_vertex_map, d_cell_vertex_counts_map, d_vertex_normals, g_n, params.multi_vertex_cells);
            int v1 = sparse_select_cell_vertex(c1, d_cell_to_vertex_map, d_cell_vertex_counts_map, d_vertex_normals, g_n, params.multi_vertex_cells);
            int v2 = sparse_select_cell_vertex(c2, d_cell_to_vertex_map, d_cell_vertex_counts_map, d_vertex_normals, g_n, params.multi_vertex_cells);
            int v3 = sparse_select_cell_vertex(c3, d_cell_to_vertex_map, d_cell_vertex_counts_map, d_vertex_normals, g_n, params.multi_vertex_cells);
            if (v0 >= 0 && v1 >= 0 && v2 >= 0 && v3 >= 0) {
                float3 pa = d_vertices[v0];
                float3 pb = d_vertices[v1];
                float3 pc = d_vertices[v2];
                float3 pd = d_vertices[v3];
                float3 quad_n = sparse_cross(sparse_operator_minus(pb, pa), sparse_operator_minus(pd, pa));
                if (sparse_dot(quad_n, quad_n) < 1e-12f) {
                    quad_n = sparse_cross(sparse_operator_minus(pc, pa), sparse_operator_minus(pd, pa));
                }
                if (sparse_dot(g_n, g_n) > 0.0f && sparse_dot(quad_n, quad_n) > 0.0f) {
                    if (sparse_dot(quad_n, g_n) < 0.0f) {
                        int tmp = v1; v1 = v3; v3 = tmp;
                    }
                }
                d_faces[4 * out_idx + 0] = v0;
                d_faces[4 * out_idx + 1] = v1;
                d_faces[4 * out_idx + 2] = v2;
                d_faces[4 * out_idx + 3] = v3;
                out_idx++;
                if (is_thin) {
                    d_faces[4 * out_idx + 0] = v0;
                    d_faces[4 * out_idx + 1] = v3;
                    d_faces[4 * out_idx + 2] = v2;
                    d_faces[4 * out_idx + 3] = v1;
                    out_idx++;
                }
            } else if (params.multi_vertex_cells) {
                atomicAdd(&d_counters->faces_skipped, 1);
            }
        }
    }

    if (ix >= 1 && iy >= 1 && iz < params.nz - 1) {
        int g0 = sparse_grid_index(ix, iy, iz, params.nx, params.ny, params.nz);
        int g1 = sparse_grid_index(ix, iy, iz + 1, params.nx, params.ny, params.nz);
        bool is_crossing = (d_sdf[g0] * d_sdf[g1] < 0.0f);
        bool is_thin = (!is_crossing && sparse_is_thin_edge(d_sdf[g0], d_sdf[g1], ix, iy, iz, ix, iy, iz + 1, d_sdf, params));
        if (is_crossing || is_thin) {
            float3 g_n = sparse_operator_plus(sparse_get_gradient(d_sdf, ix, iy, iz, params), sparse_get_gradient(d_sdf, ix, iy, iz + 1, params));
            int c0 = sparse_cell_index(ix - 1, iy - 1, iz, params.nx, params.ny, params.nz);
            int c1 = sparse_cell_index(ix, iy - 1, iz, params.nx, params.ny, params.nz);
            int c2 = sparse_cell_index(ix, iy, iz, params.nx, params.ny, params.nz);
            int c3 = sparse_cell_index(ix - 1, iy, iz, params.nx, params.ny, params.nz);
            int v0 = sparse_select_cell_vertex(c0, d_cell_to_vertex_map, d_cell_vertex_counts_map, d_vertex_normals, g_n, params.multi_vertex_cells);
            int v1 = sparse_select_cell_vertex(c1, d_cell_to_vertex_map, d_cell_vertex_counts_map, d_vertex_normals, g_n, params.multi_vertex_cells);
            int v2 = sparse_select_cell_vertex(c2, d_cell_to_vertex_map, d_cell_vertex_counts_map, d_vertex_normals, g_n, params.multi_vertex_cells);
            int v3 = sparse_select_cell_vertex(c3, d_cell_to_vertex_map, d_cell_vertex_counts_map, d_vertex_normals, g_n, params.multi_vertex_cells);
            if (v0 >= 0 && v1 >= 0 && v2 >= 0 && v3 >= 0) {
                float3 pa = d_vertices[v0];
                float3 pb = d_vertices[v1];
                float3 pc = d_vertices[v2];
                float3 pd = d_vertices[v3];
                float3 quad_n = sparse_cross(sparse_operator_minus(pb, pa), sparse_operator_minus(pd, pa));
                if (sparse_dot(quad_n, quad_n) < 1e-12f) {
                    quad_n = sparse_cross(sparse_operator_minus(pc, pa), sparse_operator_minus(pd, pa));
                }
                if (sparse_dot(g_n, g_n) > 0.0f && sparse_dot(quad_n, quad_n) > 0.0f) {
                    if (sparse_dot(quad_n, g_n) < 0.0f) {
                        int tmp = v1; v1 = v3; v3 = tmp;
                    }
                }
                d_faces[4 * out_idx + 0] = v0;
                d_faces[4 * out_idx + 1] = v1;
                d_faces[4 * out_idx + 2] = v2;
                d_faces[4 * out_idx + 3] = v3;
                out_idx++;
                if (is_thin) {
                    d_faces[4 * out_idx + 0] = v0;
                    d_faces[4 * out_idx + 1] = v3;
                    d_faces[4 * out_idx + 2] = v2;
                    d_faces[4 * out_idx + 3] = v1;
                    out_idx++;
                }
            } else if (params.multi_vertex_cells) {
                atomicAdd(&d_counters->faces_skipped, 1);
            }
        }
    }
}

struct sparse_is_active_op {
    const uint8_t* active_flags;
    __host__ __device__ bool operator()(int idx) const {
        return active_flags[idx] != 0;
    }
};

void print_sparse_gpu_memory_audit(int nx, int ny, int nz, int nbx, int nby, int nbz, int total_cells, int active_bricks, int active_cells, int total_faces) {
    size_t sdf_bytes = (size_t)nx * ny * nz * sizeof(float);
    size_t brick_flags_bytes = (size_t)nbx * nby * nbz * sizeof(uint8_t);
    size_t active_brick_list_bytes = (size_t)active_bricks * sizeof(int);
    size_t active_cell_flags_bytes = (size_t)total_cells * sizeof(uint8_t);
    size_t active_cell_list_bytes = (size_t)active_cells * sizeof(int);
    size_t cell_vertex_map_bytes = (size_t)total_cells * sizeof(int);
    size_t vertex_buffer_bytes = (size_t)active_cells * sizeof(float3);
    size_t face_buffer_bytes = (size_t)total_faces * 4 * sizeof(int);
    size_t total_est = sdf_bytes + brick_flags_bytes + active_brick_list_bytes + active_cell_flags_bytes + active_cell_list_bytes + cell_vertex_map_bytes + vertex_buffer_bytes + face_buffer_bytes;

    std::cout << "\n=== GPU Memory Estimate Before/During Extraction ===\n";
    std::cout << "SDF Grid Bytes:          " << sdf_bytes / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Brick Flags Bytes:       " << brick_flags_bytes / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Active Brick List Bytes: " << active_brick_list_bytes / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Active Cell Flags Bytes: " << active_cell_flags_bytes / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Active Cell List Bytes:  " << active_cell_list_bytes / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Cell Vertex Map Bytes:   " << cell_vertex_map_bytes / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Vertex Buffer Bytes:     " << vertex_buffer_bytes / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Face Buffer Bytes:       " << face_buffer_bytes / (1024.0 * 1024.0) << " MB\n";
    std::cout << "Total Estimated VRAM:    " << total_est / (1024.0 * 1024.0) << " MB\n====================================================\n\n";
}

DualContouringMesh CudaSparseDualContouringBackend::extract(const DenseSdfGrid& grid, DualContouringStats& stats) {
    cudaEvent_t event_start, event_stop;
    CUDA_CHECK_SPARSE(cudaEventCreate(&event_start));
    CUDA_CHECK_SPARSE(cudaEventCreate(&event_stop));

    float upload_ms = 0.0f;
    int total_vertices = grid.nx * grid.ny * grid.nz;
    float* d_sdf = nullptr;

    CUDA_CHECK_SPARSE(cudaEventRecord(event_start));
    CUDA_CHECK_SPARSE(cudaMalloc(&d_sdf, total_vertices * sizeof(float)));
    CUDA_CHECK_SPARSE(cudaMemcpy(d_sdf, grid.values.data(), total_vertices * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_SPARSE(cudaEventRecord(event_stop));
    CUDA_CHECK_SPARSE(cudaEventSynchronize(event_stop));
    CUDA_CHECK_SPARSE(cudaEventElapsedTime(&upload_ms, event_start, event_stop));

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

    CUDA_CHECK_SPARSE(cudaFree(d_sdf));
    CUDA_CHECK_SPARSE(cudaEventDestroy(event_start));
    CUDA_CHECK_SPARSE(cudaEventDestroy(event_stop));
    return mesh;
}

DualContouringMesh CudaSparseDualContouringBackend::extract_device(const DenseSdfGridDevice& grid, DualContouringStats& stats) {
    cudaEvent_t event_start, event_stop;
    CUDA_CHECK_SPARSE(cudaEventCreate(&event_start));
    CUDA_CHECK_SPARSE(cudaEventCreate(&event_stop));

    float upload_ms = 0.0f;
    float active_brick_marking_ms = 0.0f;
    float active_brick_compaction_ms = 0.0f;
    float active_cell_marking_ms = 0.0f;
    float active_cell_compaction_ms = 0.0f;
    float qef_ms = 0.0f;
    float face_count_ms = 0.0f;
    float face_prefix_sum_ms = 0.0f;
    float face_fill_ms = 0.0f;
    float download_ms = 0.0f;

    auto time_start_all = std::chrono::high_resolution_clock::now();

    int total_cells = (grid.nx - 1) * (grid.ny - 1) * (grid.nz - 1);
    int total_vertices = grid.nx * grid.ny * grid.nz;

    int nbx = (grid.nx - 1 + brick_size - 1) / brick_size;
    int nby = (grid.ny - 1 + brick_size - 1) / brick_size;
    int nbz = (grid.nz - 1 + brick_size - 1) / brick_size;
    int total_bricks = nbx * nby * nbz;

    SparseGridParams params;
    params.nx = grid.nx;
    params.ny = grid.ny;
    params.nz = grid.nz;
    params.nbx = nbx;
    params.nby = nby;
    params.nbz = nbz;
    params.brick_size = brick_size;
    params.vx = grid.vx;
    params.vy = grid.vy;
    params.vz = grid.vz;
    params.ox = grid.ox;
    params.oy = grid.oy;
    params.oz = grid.oz;
    params.normal_mode = normal_mode;
    params.multi_vertex_cells = multi_vertex_cells;

    float* d_sdf = (float*)grid.d_values;
    uint8_t* d_active_brick_flags = nullptr;
    int* d_active_brick_indices = nullptr;
    uint8_t* d_active_cell_flags = nullptr;
    int* d_active_cell_indices = nullptr;
    int* d_cell_to_vertex_map = nullptr;
    float3* d_vertices = nullptr;
    SparseDeviceCounters* d_counters = nullptr;

    CUDA_CHECK_SPARSE(cudaMalloc(&d_active_brick_flags, total_bricks * sizeof(uint8_t)));
    CUDA_CHECK_SPARSE(cudaMalloc(&d_active_brick_indices, total_bricks * sizeof(int)));
    CUDA_CHECK_SPARSE(cudaMalloc(&d_active_cell_flags, total_cells * sizeof(uint8_t)));
    CUDA_CHECK_SPARSE(cudaMalloc(&d_active_cell_indices, total_cells * sizeof(int)));
    CUDA_CHECK_SPARSE(cudaMalloc(&d_cell_to_vertex_map, total_cells * sizeof(int)));
    CUDA_CHECK_SPARSE(cudaMemset(d_cell_to_vertex_map, -1, total_cells * sizeof(int)));
    CUDA_CHECK_SPARSE(cudaMalloc(&d_counters, sizeof(SparseDeviceCounters)));
    CUDA_CHECK_SPARSE(cudaMemset(d_counters, 0, sizeof(SparseDeviceCounters)));

    int block_size = 256;
    int grid_size_bricks = (total_bricks + block_size - 1) / block_size;

    CUDA_CHECK_SPARSE(cudaEventRecord(event_start));
    mark_active_bricks_kernel<<<grid_size_bricks, block_size>>>(d_sdf, d_active_brick_flags, total_bricks, params);
    CUDA_CHECK_SPARSE(cudaEventRecord(event_stop));
    CUDA_CHECK_SPARSE(cudaEventSynchronize(event_stop));
    CUDA_CHECK_SPARSE(cudaEventElapsedTime(&active_brick_marking_ms, event_start, event_stop));

    CUDA_CHECK_SPARSE(cudaEventRecord(event_start));
    thrust::counting_iterator<int> brick_count_begin(0);
    thrust::counting_iterator<int> brick_count_end(total_bricks);
    sparse_is_active_op brick_active_pred{ d_active_brick_flags };
    int* brick_new_end = thrust::copy_if(
        thrust::device,
        brick_count_begin,
        brick_count_end,
        d_active_brick_indices,
        brick_active_pred
    );
    int active_brick_count = brick_new_end - d_active_brick_indices;
    CUDA_CHECK_SPARSE(cudaEventRecord(event_stop));
    CUDA_CHECK_SPARSE(cudaEventSynchronize(event_stop));
    CUDA_CHECK_SPARSE(cudaEventElapsedTime(&active_brick_compaction_ms, event_start, event_stop));

    CUDA_CHECK_SPARSE(cudaMemset(d_active_cell_flags, 0, total_cells * sizeof(uint8_t)));

    CUDA_CHECK_SPARSE(cudaEventRecord(event_start));
    if (active_brick_count > 0) {
        mark_active_cells_sparse_kernel<<<active_brick_count, block_size>>>(
            d_active_brick_indices,
            d_sdf,
            d_active_cell_flags,
            active_brick_count,
            params
        );
    }
    CUDA_CHECK_SPARSE(cudaEventRecord(event_stop));
    CUDA_CHECK_SPARSE(cudaEventSynchronize(event_stop));
    CUDA_CHECK_SPARSE(cudaEventElapsedTime(&active_cell_marking_ms, event_start, event_stop));

    CUDA_CHECK_SPARSE(cudaEventRecord(event_start));
    thrust::counting_iterator<int> cell_count_begin(0);
    thrust::counting_iterator<int> cell_count_end(total_cells);
    sparse_is_active_op cell_active_pred{ d_active_cell_flags };
    int* cell_new_end = thrust::copy_if(
        thrust::device,
        cell_count_begin,
        cell_count_end,
        d_active_cell_indices,
        cell_active_pred
    );
    int active_cell_count = cell_new_end - d_active_cell_indices;
    CUDA_CHECK_SPARSE(cudaEventRecord(event_stop));
    CUDA_CHECK_SPARSE(cudaEventSynchronize(event_stop));
    CUDA_CHECK_SPARSE(cudaEventElapsedTime(&active_cell_compaction_ms, event_start, event_stop));

    DualContouringMesh mesh;
    int total_faces = 0;

    if (active_cell_count > 0) {
        int grid_size_active = (active_cell_count + block_size - 1) / block_size;
        int total_vertices = active_cell_count;

        int* d_active_cell_vertex_counts = nullptr;
        int* d_scanned_vertex_counts = nullptr;
        int* d_cell_vertex_counts_map = nullptr;
        float3* d_vertex_normals = nullptr;

        if (params.multi_vertex_cells) {
            CUDA_CHECK_SPARSE(cudaMalloc(&d_active_cell_vertex_counts, active_cell_count * sizeof(int)));
            CUDA_CHECK_SPARSE(cudaMalloc(&d_scanned_vertex_counts, active_cell_count * sizeof(int)));
            CUDA_CHECK_SPARSE(cudaMalloc(&d_cell_vertex_counts_map, total_cells * sizeof(int)));
            CUDA_CHECK_SPARSE(cudaMemset(d_cell_vertex_counts_map, 0, total_cells * sizeof(int)));

            count_cell_vertices_sparse_kernel<<<grid_size_active, block_size>>>(
                d_active_cell_indices,
                d_sdf,
                nullptr,
                d_active_cell_vertex_counts,
                active_cell_count,
                params,
                d_counters
            );
            thrust::exclusive_scan(thrust::device, d_active_cell_vertex_counts, d_active_cell_vertex_counts + active_cell_count, d_scanned_vertex_counts);

            int last_v_count = 0;
            int last_v_scanned = 0;
            CUDA_CHECK_SPARSE(cudaMemcpy(&last_v_count, d_active_cell_vertex_counts + active_cell_count - 1, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK_SPARSE(cudaMemcpy(&last_v_scanned, d_scanned_vertex_counts + active_cell_count - 1, sizeof(int), cudaMemcpyDeviceToHost));
            total_vertices = last_v_scanned + last_v_count;
        }

        CUDA_CHECK_SPARSE(cudaMalloc(&d_vertices, total_vertices * sizeof(float3)));
        if (params.multi_vertex_cells && total_vertices > 0) {
            CUDA_CHECK_SPARSE(cudaMalloc(&d_vertex_normals, total_vertices * sizeof(float3)));
        }

        if (total_vertices > 0) {
            CUDA_CHECK_SPARSE(cudaEventRecord(event_start));
            compute_vertices_sparse_kernel<<<grid_size_active, block_size>>>(
                d_active_cell_indices,
                d_sdf,
                nullptr,
                d_vertices,
                d_vertex_normals,
                d_cell_to_vertex_map,
                d_cell_vertex_counts_map,
                d_scanned_vertex_counts,
                d_active_cell_vertex_counts,
                active_cell_count,
                params,
                d_counters
            );
            CUDA_CHECK_SPARSE(cudaEventRecord(event_stop));
            CUDA_CHECK_SPARSE(cudaEventSynchronize(event_stop));
            CUDA_CHECK_SPARSE(cudaEventElapsedTime(&qef_ms, event_start, event_stop));
        }

        int* d_active_cell_face_counts = nullptr;
        int* d_scanned_face_counts = nullptr;
        int* d_faces = nullptr;

        CUDA_CHECK_SPARSE(cudaMalloc(&d_active_cell_face_counts, active_cell_count * sizeof(int)));
        CUDA_CHECK_SPARSE(cudaMalloc(&d_scanned_face_counts, active_cell_count * sizeof(int)));

        CUDA_CHECK_SPARSE(cudaEventRecord(event_start));
        count_sparse_faces_kernel<<<grid_size_active, block_size>>>(
            d_active_cell_indices,
            d_sdf,
            d_cell_to_vertex_map,
            d_cell_vertex_counts_map,
            d_vertex_normals,
            d_active_cell_face_counts,
            active_cell_count,
            params
        );
        CUDA_CHECK_SPARSE(cudaEventRecord(event_stop));
        CUDA_CHECK_SPARSE(cudaEventSynchronize(event_stop));
        CUDA_CHECK_SPARSE(cudaEventElapsedTime(&face_count_ms, event_start, event_stop));

        CUDA_CHECK_SPARSE(cudaEventRecord(event_start));
        thrust::exclusive_scan(thrust::device, d_active_cell_face_counts, d_active_cell_face_counts + active_cell_count, d_scanned_face_counts);
        CUDA_CHECK_SPARSE(cudaEventRecord(event_stop));
        CUDA_CHECK_SPARSE(cudaEventSynchronize(event_stop));
        CUDA_CHECK_SPARSE(cudaEventElapsedTime(&face_prefix_sum_ms, event_start, event_stop));

        int last_count = 0;
        int last_scanned = 0;
        CUDA_CHECK_SPARSE(cudaMemcpy(&last_count, d_active_cell_face_counts + active_cell_count - 1, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK_SPARSE(cudaMemcpy(&last_scanned, d_scanned_face_counts + active_cell_count - 1, sizeof(int), cudaMemcpyDeviceToHost));
        total_faces = last_scanned + last_count;

        CUDA_CHECK_SPARSE(cudaEventRecord(event_start));
        if (total_faces > 0 && total_vertices > 0) {
            CUDA_CHECK_SPARSE(cudaMalloc(&d_faces, total_faces * 4 * sizeof(int)));
            emit_sparse_faces_kernel<<<grid_size_active, block_size>>>(
                d_active_cell_indices,
                d_sdf,
                d_cell_to_vertex_map,
                d_cell_vertex_counts_map,
                d_vertices,
                d_vertex_normals,
                d_active_cell_face_counts,
                d_scanned_face_counts,
                d_faces,
                active_cell_count,
                params,
                d_counters
            );
        }
        CUDA_CHECK_SPARSE(cudaEventRecord(event_stop));
        CUDA_CHECK_SPARSE(cudaEventSynchronize(event_stop));
        CUDA_CHECK_SPARSE(cudaEventElapsedTime(&face_fill_ms, event_start, event_stop));

        CUDA_CHECK_SPARSE(cudaEventRecord(event_start));
        mesh.vertices.resize(total_vertices * 3);
        if (total_vertices > 0) {
            CUDA_CHECK_SPARSE(cudaMemcpy(mesh.vertices.data(), d_vertices, total_vertices * sizeof(float3), cudaMemcpyDeviceToHost));
        }

        if (total_faces > 0 && total_vertices > 0) {
            mesh.faces.resize(total_faces * 4);
            CUDA_CHECK_SPARSE(cudaMemcpy(mesh.faces.data(), d_faces, total_faces * 4 * sizeof(int), cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK_SPARSE(cudaEventRecord(event_stop));
        CUDA_CHECK_SPARSE(cudaEventSynchronize(event_stop));
        CUDA_CHECK_SPARSE(cudaEventElapsedTime(&download_ms, event_start, event_stop));

        if (d_faces) CUDA_CHECK_SPARSE(cudaFree(d_faces));
        CUDA_CHECK_SPARSE(cudaFree(d_active_cell_face_counts));
        CUDA_CHECK_SPARSE(cudaFree(d_scanned_face_counts));
        if (d_vertices) CUDA_CHECK_SPARSE(cudaFree(d_vertices));
        if (d_vertex_normals) CUDA_CHECK_SPARSE(cudaFree(d_vertex_normals));
        if (d_active_cell_vertex_counts) CUDA_CHECK_SPARSE(cudaFree(d_active_cell_vertex_counts));
        if (d_scanned_vertex_counts) CUDA_CHECK_SPARSE(cudaFree(d_scanned_vertex_counts));
        if (d_cell_vertex_counts_map) CUDA_CHECK_SPARSE(cudaFree(d_cell_vertex_counts_map));
    }

    SparseDeviceCounters host_counters;
    CUDA_CHECK_SPARSE(cudaMemcpy(&host_counters, d_counters, sizeof(SparseDeviceCounters), cudaMemcpyDeviceToHost));

    print_sparse_gpu_memory_audit(grid.nx, grid.ny, grid.nz, nbx, nby, nbz, total_cells, active_brick_count, active_cell_count, total_faces);

    CUDA_CHECK_SPARSE(cudaFree(d_active_brick_flags));
    CUDA_CHECK_SPARSE(cudaFree(d_active_brick_indices));
    CUDA_CHECK_SPARSE(cudaFree(d_active_cell_flags));
    CUDA_CHECK_SPARSE(cudaFree(d_active_cell_indices));
    CUDA_CHECK_SPARSE(cudaFree(d_cell_to_vertex_map));
    CUDA_CHECK_SPARSE(cudaFree(d_counters));

    CUDA_CHECK_SPARSE(cudaEventDestroy(event_start));
    CUDA_CHECK_SPARSE(cudaEventDestroy(event_stop));

    auto time_end_all = std::chrono::high_resolution_clock::now();
    float total_ms = std::chrono::duration<float, std::milli>(time_end_all - time_start_all).count();

    stats.backend = params.multi_vertex_cells ? "cuda-sparse-mvdc" : "cuda-sparse";
    stats.nx = grid.nx;
    stats.ny = grid.ny;
    stats.nz = grid.nz;
    stats.total_cells = total_cells;
    stats.active_cells = active_cell_count;
    stats.total_bricks = total_bricks;
    stats.active_bricks = active_brick_count;
    stats.upload_ms = upload_ms;
    stats.active_brick_marking_ms = active_brick_marking_ms;
    stats.active_brick_compaction_ms = active_brick_compaction_ms;
    stats.active_cell_marking_ms = active_cell_marking_ms;
    stats.active_cell_compaction_ms = active_cell_compaction_ms;
    stats.marking_ms = active_brick_marking_ms + active_cell_marking_ms;
    stats.compaction_ms = active_brick_compaction_ms + active_cell_compaction_ms;
    stats.qef_ms = qef_ms;
    stats.face_count_ms = face_count_ms;
    stats.face_prefix_sum_ms = face_prefix_sum_ms;
    stats.face_fill_ms = face_fill_ms;
    stats.face_emission_ms = face_count_ms + face_prefix_sum_ms + face_fill_ms;
    stats.download_ms = download_ms;
    stats.total_ms = total_ms;
    stats.vertex_count = mesh.vertices.size() / 3;
    stats.face_count = mesh.faces.size() / 4;
    stats.qef_fallback_count = host_counters.qef_fallbacks;
    stats.clamp_count = host_counters.out_of_cell_clamps;
    stats.invalid_count = host_counters.invalid_count;
    stats.ambiguous_cells = host_counters.ambiguous_cells;
    stats.multi_vertex_cells = host_counters.multi_vertex_cells;
    stats.one_vertex_cells = host_counters.one_vertex_cells;
    stats.two_vertex_cells = host_counters.two_vertex_cells;
    stats.split_rejection_count = host_counters.split_rejections;
    stats.bad_qef_count = host_counters.bad_qef_count;
    stats.faces_skipped_due_to_missing_cluster = host_counters.faces_skipped;

    return mesh;
}
