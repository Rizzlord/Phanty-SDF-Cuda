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

DualContouringMesh postprocess_mesh(const DualContouringMesh& mesh, bool close_holes, bool remove_floaters) {
    if (mesh.vertices.empty() || mesh.faces.empty() || (!close_holes && !remove_floaters)) {
        return mesh;
    }

    int num_vertices = (int)(mesh.vertices.size() / 3);
    int num_faces = (int)(mesh.faces.size() / 4);

    struct UndirectedEdge {
        int u, v; // u < v
        int face_idx;
        int edge_idx; // 0, 1, 2, or 3
        bool operator<(const UndirectedEdge& o) const {
            if (u != o.u) return u < o.u;
            return v < o.v;
        }
    };

    std::vector<UndirectedEdge> all_edges;
    all_edges.reserve(num_faces * 4);
    for (int f = 0; f < num_faces; ++f) {
        for (int e = 0; e < 4; ++e) {
            int u = mesh.faces[4 * f + e];
            int v = mesh.faces[4 * f + (e + 1) % 4];
            if (u == v) continue;
            UndirectedEdge edge;
            edge.u = std::min(u, v);
            edge.v = std::max(u, v);
            edge.face_idx = f;
            edge.edge_idx = e;
            all_edges.push_back(edge);
        }
    }

    std::sort(all_edges.begin(), all_edges.end());

    std::vector<int> face_neighbors(num_faces * 4, -1);
    std::vector<int> neighbor_counts(num_faces, 0);

    struct DirectedBoundaryEdge {
        int u, v;
        int face_idx;
    };
    std::vector<DirectedBoundaryEdge> boundary_edges;

    size_t i = 0;
    while (i < all_edges.size()) {
        size_t j = i + 1;
        while (j < all_edges.size() && all_edges[j].u == all_edges[i].u && all_edges[j].v == all_edges[i].v) {
            j++;
        }

        size_t count = j - i;
        if (count == 1) {
            int f = all_edges[i].face_idx;
            int e = all_edges[i].edge_idx;
            int u = mesh.faces[4 * f + e];
            int v = mesh.faces[4 * f + (e + 1) % 4];
            boundary_edges.push_back({u, v, f});
        } else {
            for (size_t x = i; x < j; ++x) {
                for (size_t y = x + 1; y < j; ++y) {
                    int f1 = all_edges[x].face_idx;
                    int f2 = all_edges[y].face_idx;
                    if (neighbor_counts[f1] < 4) face_neighbors[4 * f1 + neighbor_counts[f1]++] = f2;
                    if (neighbor_counts[f2] < 4) face_neighbors[4 * f2 + neighbor_counts[f2]++] = f1;
                }
            }
        }
        i = j;
    }

    std::vector<int> kept_faces;
    kept_faces.reserve(num_faces);
    std::vector<int> face_component(num_faces, 0);
    int largest_comp_idx = 0;

    if (remove_floaters) {
        std::fill(face_component.begin(), face_component.end(), -1);
        std::vector<int> component_sizes;
        std::vector<int> queue(num_faces);

        int num_components = 0;
        for (int f = 0; f < num_faces; ++f) {
            if (face_component[f] != -1) continue;

            int comp_idx = num_components++;
            int q_head = 0;
            int q_tail = 0;

            queue[q_tail++] = f;
            face_component[f] = comp_idx;
            int comp_size = 0;

            while (q_head < q_tail) {
                int curr = queue[q_head++];
                comp_size++;

                int n_start = 4 * curr;
                int n_count = neighbor_counts[curr];
                for (int n = 0; n < n_count; ++n) {
                    int neighbor = face_neighbors[n_start + n];
                    if (neighbor != -1 && face_component[neighbor] == -1) {
                        face_component[neighbor] = comp_idx;
                        queue[q_tail++] = neighbor;
                    }
                }
            }
            component_sizes.push_back(comp_size);
        }

        largest_comp_idx = -1;
        int max_size = -1;
        for (int c = 0; c < num_components; ++c) {
            if (component_sizes[c] > max_size) {
                max_size = component_sizes[c];
                largest_comp_idx = c;
            }
        }

        for (int f = 0; f < num_faces; ++f) {
            if (face_component[f] == largest_comp_idx) {
                kept_faces.push_back(f);
            }
        }
    } else {
        for (int f = 0; f < num_faces; ++f) {
            kept_faces.push_back(f);
        }
    }

    std::vector<int> active_faces;
    active_faces.reserve(kept_faces.size() * 4);
    for (int f : kept_faces) {
        for (int k = 0; k < 4; ++k) {
            active_faces.push_back(mesh.faces[4 * f + k]);
        }
    }

    if (close_holes && !boundary_edges.empty()) {
        std::vector<std::vector<int>> adj_boundary(num_vertices);
        for (const auto& edge : boundary_edges) {
            if (!remove_floaters || (remove_floaters && face_component[edge.face_idx] == largest_comp_idx)) {
                adj_boundary[edge.u].push_back(edge.v);
            }
        }

        for (int v_start = 0; v_start < num_vertices; ++v_start) {
            while (!adj_boundary[v_start].empty()) {
                int curr = v_start;
                std::vector<int> loop;
                loop.push_back(curr);

                bool closed = false;
                while (true) {
                    if (adj_boundary[curr].empty()) {
                        break;
                    }
                    int next = adj_boundary[curr].back();
                    adj_boundary[curr].pop_back();

                    if (next == v_start) {
                        closed = true;
                        break;
                    }

                    auto it = std::find(loop.begin(), loop.end(), next);
                    if (it != loop.end()) {
                        std::vector<int> sub_loop(it, loop.end());
                        loop.erase(it, loop.end());
                        if (sub_loop.size() >= 3 && sub_loop.size() <= 64) {
                            std::vector<int> rev_loop(sub_loop.rbegin(), sub_loop.rend());
                            if (rev_loop.size() == 3) {
                                active_faces.push_back(rev_loop[2]);
                                active_faces.push_back(rev_loop[1]);
                                active_faces.push_back(rev_loop[0]);
                                active_faces.push_back(rev_loop[0]);
                            } else if (rev_loop.size() == 4) {
                                active_faces.push_back(rev_loop[3]);
                                active_faces.push_back(rev_loop[2]);
                                active_faces.push_back(rev_loop[1]);
                                active_faces.push_back(rev_loop[0]);
                            } else {
                                for (size_t idx = 1; idx < rev_loop.size() - 1; ++idx) {
                                    active_faces.push_back(rev_loop[0]);
                                    active_faces.push_back(rev_loop[idx]);
                                    active_faces.push_back(rev_loop[idx + 1]);
                                    active_faces.push_back(rev_loop[idx + 1]);
                                }
                            }
                        }
                        curr = next;
                        loop.push_back(curr);
                        continue;
                    }

                    loop.push_back(next);
                    curr = next;
                }

                if (closed && loop.size() >= 3 && loop.size() <= 64) {
                    std::vector<int> rev_loop(loop.rbegin(), loop.rend());
                    if (rev_loop.size() == 3) {
                        active_faces.push_back(rev_loop[2]);
                        active_faces.push_back(rev_loop[1]);
                        active_faces.push_back(rev_loop[0]);
                        active_faces.push_back(rev_loop[0]);
                    } else if (rev_loop.size() == 4) {
                        active_faces.push_back(rev_loop[3]);
                        active_faces.push_back(rev_loop[2]);
                        active_faces.push_back(rev_loop[1]);
                        active_faces.push_back(rev_loop[0]);
                    } else {
                        for (size_t idx = 1; idx < rev_loop.size() - 1; ++idx) {
                            active_faces.push_back(rev_loop[0]);
                            active_faces.push_back(rev_loop[idx]);
                            active_faces.push_back(rev_loop[idx + 1]);
                            active_faces.push_back(rev_loop[idx + 1]);
                        }
                    }
                }
            }
        }
    }

    std::vector<int> old_to_new_vertex(num_vertices, -1);
    DualContouringMesh processed_mesh;
    processed_mesh.vertices.reserve(mesh.vertices.size());
    processed_mesh.faces.reserve(active_faces.size());

    for (int old_v : active_faces) {
        if (old_to_new_vertex[old_v] == -1) {
            old_to_new_vertex[old_v] = (int)(processed_mesh.vertices.size() / 3);
            processed_mesh.vertices.push_back(mesh.vertices[3 * old_v + 0]);
            processed_mesh.vertices.push_back(mesh.vertices[3 * old_v + 1]);
            processed_mesh.vertices.push_back(mesh.vertices[3 * old_v + 2]);
        }
        processed_mesh.faces.push_back(old_to_new_vertex[old_v]);
    }

    return processed_mesh;
}

