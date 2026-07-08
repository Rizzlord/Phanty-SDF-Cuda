#include "ui.h"
#include "contouring.h"
#include "Cell.h"
#include "vertex_refinement.h"
#include "hermite_update.h"


#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>
#include <imgui.h>
#include <glm/glm.hpp>
#include <unordered_map>



int outer_iteration_counter = 0;


void step_cell_inner_iteration(Cell& cell, const ContouringOptions& options) {
    if (!cell.has_vertex) return;   
    cell.prev_vertex = cell.vertex;
    refine_vertex_from_face_intersections(cell);
    if (!cell.closest_points_info.empty()) {
        cell.minimize_qef(
            options.mu,
            options.dc_weight,
            options.sphere_weight,
            options.svd_threshold,
            options.verbose
        );
    }
}


void start_outer_iteration(
    std::vector<Cell>& cells,
    const Eigen::VectorXd& S,
    const Eigen::MatrixXd& GV,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    int resX,
    int resY,
    int resZ,
    const ContouringOptions& opts)
{

    // Triangulate F
    Eigen::MatrixXi TriF = triangulate_v1(F, V);

    // Optimize the triangulation of each quad based on the distance to spheres
    optimize_triangulation(S, GV, V, F, TriF, resX, resY, resZ);

    if (opts.verbose) { std::cout << "[contouring] Assigning spheres..." << std::endl; }
    assign_spheres_to_cells(S, GV, V, F, cells, resX, resY, resZ, opts.batch_size, TriF);
    if (opts.verbose) { std::cout << "[contouring] Computing intersections..." << std::endl; }
    compute_face_cell_intersections(cells, resX, resY, resZ, opts.new_face_pos_weight);

    if (opts.hermite_update && outer_iteration_counter > 0) {
        if (opts.verbose) { std::cout << "[contouring] Updating Hermite points and normals..." << std::endl; }
        update_hermite_points_and_normals(cells, resX, resY, resZ, opts.new_hermite_normal_weight, opts.new_hermite_pos_weight, true, opts.verbose);
    }
}


void finish_outer_iteration(
    std::vector<Cell>& cells,
    const Eigen::VectorXd& S,
    const Eigen::MatrixXd& GV,
    int resX,
    int resY,
    int resZ,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    const ContouringOptions opts
) {
    // Clean all cells
    #pragma omp parallel for
    for (Cell& c : cells) {
        // Check that the cell has finished all inner iterations (options.inner_iters)
        // If it has not, finish them now
        while (c.has_vertex && c.inner_iter < opts.inner_iters) {
            step_cell_inner_iteration(c, opts);
            c.inner_iter++;
        }

        c.clean();
    }

    // Reextract mesh
    extract_mesh_from_cells(cells, S, GV, resX, resY, resZ, V, F);

    auto* mesh_outer_iter = polyscope::registerSurfaceMesh(
        "outer_iteration_" + std::to_string(outer_iteration_counter), V, F);
    mesh_outer_iter->setSmoothShade(false);
    mesh_outer_iter->setEnabled(false);  

    outer_iteration_counter++;

    show_total_energy(cells, opts.verbose);
}


void update_current_cell_visualization(
    std::vector<Cell>* cells,
    int sel)
{
    if (cells && sel >= 0 && sel < (int)cells->size()) {
        (*cells)[sel].visualize();
    }
}

// Helper: toggle selection of a Hermite point. Clicking a Hermite point will
// select it (color it green and show associated spheres). Clicking again
// deselects it (restore default color and hide spheres).
static void handleHermitePointPick(const polyscope::PickResult& selPick) {
    if (!selPick.isHit || selPick.structure == nullptr) return;
    std::string sname = selPick.structure->getName();
    if (!(sname.find("Cell_") == 0 && sname.find("Hermite points") != std::string::npos)) return;

    // track per-structure selected local index (-1 == none)
    static std::unordered_map<std::string,int> selectedMap;

    // Parse cell indices from the name: "Cell_ix_iy_iz_Hermite points"
    std::string prefix = sname.substr(0, sname.find("Hermite points"));
    if (!prefix.empty() && prefix.back() == '_') prefix.pop_back();
    std::vector<int> idxs;
    size_t pos = prefix.find("_");
    while (pos != std::string::npos) {
        size_t next = prefix.find("_", pos + 1);
        std::string token;
        if (next == std::string::npos) token = prefix.substr(pos + 1);
        else token = prefix.substr(pos + 1, next - pos - 1);
        try { idxs.push_back(std::stoi(token)); } catch(...) {}
        pos = next;
    }
    if (idxs.size() < 3) return;
    int ix = idxs[0], iy = idxs[1], iz = idxs[2];

    std::vector<Cell>* cellsPtr = getGeneratedCells();
    if (!cellsPtr) return;

    for (size_t ci = 0; ci < cellsPtr->size(); ++ci) {
        Cell& c = (*cellsPtr)[ci];
        if (c.ix != ix || c.iy != iy || c.iz != iz) continue;

        // map local pick index -> hermite edge index
        size_t localIdx = static_cast<size_t>(selPick.localIndex);
        size_t cur = 0;
        int selectedEdgeIdx = -1;
        for (const auto& kv : c.hermite_positions) {
            if (cur == localIdx) { selectedEdgeIdx = kv.first; break; }
            cur++;
        }

        std::string basePrefix = "Cell_" + std::to_string(c.ix) + "_" + std::to_string(c.iy) + "_" + std::to_string(c.iz) + "_";

        // Point cloud colors
        try {
            polyscope::PointCloud* pc = polyscope::getPointCloud(sname);
            if (!pc) break;

            size_t nPts = pc->nPoints();
            if (localIdx >= nPts) break;

            // get or create color quantity
            polyscope::PointCloudColorQuantity* colQ = nullptr;
            auto existing = pc->getQuantity("Hermite color");
            std::vector<glm::vec3> colors(nPts, glm::vec3(0.5f, 0.0f, 0.5f));
            if (existing) {
                colQ = dynamic_cast<polyscope::PointCloudColorQuantity*>(existing);
                if (colQ) {
                    for (size_t i = 0; i < nPts; ++i) colors[i] = colQ->colors.getValue(i);
                }
            }

            // Determine previously selected index for this structure
            int prevSel = -1;
            auto itSel = selectedMap.find(sname);
            if (itSel != selectedMap.end()) prevSel = itSel->second;

            // Selecting a new point: clear previous selection color if any
            if (prevSel >= 0 && prevSel < (int)nPts) {
                colors[prevSel] = glm::vec3(0.5f, 0.0f, 0.5f);
            }
            // set new selected point color (violet)
            colors[localIdx] = glm::vec3(0.5f, 0.0f, 0.5f);
            if (colQ) colQ->updateData(colors);
            else pc->addColorQuantity("Hermite color", colors)->setEnabled(true);
            selectedMap[sname] = (int)localIdx;

            // remove any previous sphere visuals
            polyscope::removeStructure(basePrefix + "Hermite Edge Spheres", false);
            polyscope::removeStructure(basePrefix + "Hermite Edge Sphere Centers", false);

            // show spheres for this edge if present
            if (selectedEdgeIdx >= 0) {
                auto it = c.hermite_spheres.find(selectedEdgeIdx);
                if (it != c.hermite_spheres.end() && !it->second.empty()) {
                    std::cout << "[UI] Selected Hermite point on " << sname << ", # assigned spheres = " << it->second.size() << std::endl;
                    const std::vector<Sphere>& ss = it->second;
                    Eigen::MatrixXd P(ss.size(), 3);
                    Eigen::VectorXd R(ss.size());
                    Eigen::MatrixXd centerColors(ss.size(), 3);
                    for (size_t si = 0; si < ss.size(); ++si) {
                        P.row(si) = ss[si].center.transpose();
                        R(si) = ss[si].radius;
                        centerColors.row(si) = Eigen::Vector3d(0.5, 0.0, 0.5);   // Purple
                    }

                    auto* pc_spheres_edge = polyscope::registerPointCloud(basePrefix + "Hermite Edge Spheres", P);
                    pc_spheres_edge->addScalarQuantity("radius", R)->setEnabled(true);
                    pc_spheres_edge->setPointRadiusQuantity("radius");
                    pc_spheres_edge->setPointRenderMode(polyscope::PointRenderMode::Sphere);
                    pc_spheres_edge->setTransparency(0.3);

                    auto* pc_centers_edge = polyscope::registerPointCloud(basePrefix + "Hermite Edge Sphere Centers", P);
                    pc_centers_edge->addColorQuantity("Center Color", centerColors)->setEnabled(true);
                    pc_centers_edge->setPointRadius(0.05);
                    pc_centers_edge->setPointRenderMode(polyscope::PointRenderMode::Sphere);
                } else {
                    std::cout << "[UI] Selected Hermite point on " << sname << ", but no assigned spheres." << std::endl;
                }
            }

        } catch(...) {
            // ignore Polyscope errors
        }

        break; // processed the matching cell
    }
}


void install_cell_visualizer_UI(
    const Eigen::VectorXd* S_ptr,
    const Eigen::MatrixXd* GV_ptr,
    int resX, int resY, int resZ,
    const ContouringOptions* opts,
    Eigen::MatrixXd* Vmesh_ptr,
    Eigen::MatrixXi* Fmesh_ptr)
{
    polyscope::state::userCallback = [S_ptr, GV_ptr, resX, resY, resZ, opts, Vmesh_ptr, Fmesh_ptr]() {
        static int sel = 0;
        static int prevSel = -1;
        std::vector<Cell>* cells = getGeneratedCells();
        int nCells = cells ? (int)cells->size() : 0;

        auto removeCellStructures = [&](int idx) {
            if (idx < 0 || !cells) return;
            const Cell& c = (*cells)[idx];
            std::string prefix = "Cell_" + std::to_string(c.ix) + "_" + std::to_string(c.iy) + "_" + std::to_string(c.iz) + "_";
            const std::vector<std::string> names = {
                "Hermite points",
                "Corners",
                "Cell vertex",
                "Cell edges",
                "Assigned Spheres",
                "Hermite Edge Spheres",
                "Hermite Edge Sphere Centers",
                "Sphere Centers",
                "Face Intersections",
                "Cell Mesh",
                "Q Points",
                "T Points"
            };
            for (const auto& n : names) {
                polyscope::removeStructure(prefix + n, /*errorIfAbsent=*/false);
            }
        };

        ImGui::Text("Cell visualizer");
        ImGui::Text("Total cells: %d", nCells);

        if (nCells == 0) {
            if (ImGui::Button("Refresh cells")) {
                // no-op: cells are generated during contouring; user can re-run contouring externally
            }
            return;
        }


        // If a Hermite point is clicked, toggle selection and show/hide associated spheres
        {
            polyscope::PickResult selPick2 = polyscope::getSelection();
            handleHermitePointPick(selPick2);
        }

        ImGui::PushItemWidth(140 * polyscope::options::uiScale);
        if (ImGui::InputInt("Cell index", &sel)) {
            sel = std::max(0, std::min(sel, nCells - 1));
            if (sel != prevSel) {
                removeCellStructures(prevSel);
                (*cells)[sel].visualize();
                prevSel = sel;
            }
        }
        ImGui::PopItemWidth();

        // quick jump to first/last
        if (ImGui::Button("First")) {
            if (0 != prevSel) removeCellStructures(prevSel);
            sel = 0;
            (*cells)[sel].visualize();
            prevSel = sel;
        }
        ImGui::SameLine();
        if (ImGui::Button("Last")) {
            if ((nCells - 1) != prevSel) removeCellStructures(prevSel);
            sel = nCells - 1;
            (*cells)[sel].visualize();
            prevSel = sel;
        }

        ImGui::SameLine();
        if (ImGui::Button("Clear all cell visuals")) {
            // remove all registered per-cell structures and clear stored cells
            if (cells) {
                for (int i = 0; i < (int)cells->size(); ++i) removeCellStructures(i);
            }
            clearGeneratedCells();
        }

        if (ImGui::Button("Start outer iter (all cells)")) {
            if (S_ptr && GV_ptr && opts && Vmesh_ptr && Fmesh_ptr) {
                std::vector<Cell>* cells = getGeneratedCells();
                if (cells) {
                    start_outer_iteration(*cells, *S_ptr, *GV_ptr, *Vmesh_ptr, *Fmesh_ptr, resX, resY, resZ, *opts);
                    update_current_cell_visualization(cells, sel);
                }
            }
        }

        // Step one inner iteration across all stored cells and update the 'Our Method' mesh
        if (ImGui::Button("Step inner iter (cell)")) {
            if (S_ptr && GV_ptr && opts && Vmesh_ptr && Fmesh_ptr) {
                std::vector<Cell>* cells = getGeneratedCells();
                if (cells && sel >= 0 && sel < (int)cells->size()) {
                    // If the selected cell has already reached max inner iters, do nothing
                    if ((*cells)[sel].inner_iter >= opts->inner_iters) return;
                    step_cell_inner_iteration((*cells)[sel], *opts);
                    (*cells)[sel].inner_iter++;

                    update_current_cell_visualization(cells, sel);
                }
            }
        }

        ImGui::SameLine();
        // Input for advancing N inner iterations on the selected cell
        static int advanceN_inner = 1;
        ImGui::PushItemWidth(80 * polyscope::options::uiScale);
        if (ImGui::InputInt("N", &advanceN_inner)) {
            if (advanceN_inner < 1) advanceN_inner = 1;
        }
        ImGui::PopItemWidth();

        ImGui::SameLine();
        if (ImGui::Button("Advance N inner iters (cell)")) {
            if (S_ptr && GV_ptr && opts && Vmesh_ptr && Fmesh_ptr) {
                std::vector<Cell>* cells = getGeneratedCells();
                if (cells && sel >= 0 && sel < (int)cells->size()) {
                    Cell& c = (*cells)[sel];
                    if (c.has_vertex) {
                        int remaining = opts->inner_iters - c.inner_iter;
                        int toRun = std::min(advanceN_inner, remaining);
                        for (int it = 0; it < toRun; ++it) {
                            Eigen::Vector3d prev = c.vertex;
                            step_cell_inner_iteration(c, *opts);
                            c.inner_iter++;
                            if ((c.vertex - prev).norm() < 1e-6) break;
                        }
                    }
                    update_current_cell_visualization(cells, sel);
                }
            }
        }

        

        if (ImGui::Button("Finish inner iters (cell)")) {
            if (S_ptr && GV_ptr && opts && Vmesh_ptr && Fmesh_ptr) {
                std::vector<Cell>* cells = getGeneratedCells();
                if (cells && sel >= 0 && sel < (int)cells->size()) {
                    Cell& c = (*cells)[sel];
                    if (c.has_vertex) {
                        while (c.inner_iter < opts->inner_iters) {
                            Eigen::Vector3d prev = c.vertex;
                            step_cell_inner_iteration(c, *opts);
                            c.inner_iter++;
                            if ((c.vertex - prev).norm() < 1e-6) break;
                        }
                    }

                    update_current_cell_visualization(cells, sel);
                }
            }
        }

        ImGui::SameLine();
        if (ImGui::Button("Finish inner iters (all cells)")) {
            if (S_ptr && GV_ptr && opts && Vmesh_ptr && Fmesh_ptr) {
                std::vector<Cell>* cells = getGeneratedCells();
                if (cells) {
                    for (auto &c : *cells) {
                        if (!c.has_vertex) continue;
                        while (c.inner_iter < opts->inner_iters) {
                            Eigen::Vector3d prev = c.vertex;
                            step_cell_inner_iteration(c, *opts);
                            c.inner_iter++;
                            if ((c.vertex - prev).norm() < 1e-6) break;
                        }
                    }

                    // Update the visualization of the currently selected cell
                    update_current_cell_visualization(cells, sel);
                }
            }
        }

        if (ImGui::Button("Finish outer iter (all cells)")) {
            if (S_ptr && GV_ptr && opts && Vmesh_ptr && Fmesh_ptr) {
                std::vector<Cell>* cells = getGeneratedCells();
                if (cells) {
                    std::cout << "[UI] Finishing outer iteration " << outer_iteration_counter << "." << std::endl;
                    finish_outer_iteration(*cells, *S_ptr, *GV_ptr, resX, resY, resZ, *Vmesh_ptr, *Fmesh_ptr, *opts);
                    update_current_cell_visualization(cells, sel);
                }
            }
        }

        if (ImGui::Button("Advance outer iter")) {
            if (S_ptr && GV_ptr && opts && Vmesh_ptr && Fmesh_ptr) {
                std::vector<Cell>* cells = getGeneratedCells();
                if (cells) {
                    std::cout << "[UI] Advancing one outer iteration: " << outer_iteration_counter << "." << std::endl;
                    start_outer_iteration(*cells, *S_ptr, *GV_ptr, *Vmesh_ptr, *Fmesh_ptr, resX, resY, resZ, *opts);
                    finish_outer_iteration(*cells, *S_ptr, *GV_ptr, resX, resY, resZ, *Vmesh_ptr, *Fmesh_ptr, *opts);
                    update_current_cell_visualization(cells, sel);
                }
            }
        }

        ImGui::SameLine();
        static int advanceM_outer = 1;
        ImGui::PushItemWidth(80 * polyscope::options::uiScale);
        if (ImGui::InputInt("M", &advanceM_outer)) {
            if (advanceM_outer < 1) advanceM_outer = 1;
        }
        ImGui::PopItemWidth();

        ImGui::SameLine();
        if (ImGui::Button("Advance M outer iters (all cells)")) {
            if (S_ptr && GV_ptr && opts && Vmesh_ptr && Fmesh_ptr) {
                std::vector<Cell>* cells = getGeneratedCells();
                if (cells) {
                    for (int it = 0; it < advanceM_outer; ++it) {
                        start_outer_iteration(*cells, *S_ptr, *GV_ptr, *Vmesh_ptr, *Fmesh_ptr, resX, resY, resZ, *opts);
                        finish_outer_iteration(*cells, *S_ptr, *GV_ptr, resX, resY, resZ, *Vmesh_ptr, *Fmesh_ptr, *opts);
                        std::cout << "[UI] Completed outer iteration " << (it + 1) << " of " << advanceM_outer << "." << std::endl;
                        std::cout << "[UI] Now at outer iteration " << outer_iteration_counter << "." << std::endl;
                    }
                    update_current_cell_visualization(cells, sel);
                }
            }
        }
    };
}