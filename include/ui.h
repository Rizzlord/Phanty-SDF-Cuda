#pragma once

#include <Eigen/Core>

struct ContouringOptions;
class Cell;

std::vector<Cell>* getGeneratedCells();
void clearGeneratedCells();
extern std::vector<Cell>* globalCellsPtr;


void update_current_cell_visualization(
    std::vector<Cell>* cells,
    int sel
);

void install_cell_visualizer_UI(
    const Eigen::VectorXd* S_ptr,
    const Eigen::MatrixXd* GV_ptr,
    int resX, int resY, int resZ,
    const ContouringOptions* opts,
    Eigen::MatrixXd* Vmesh_ptr,
    Eigen::MatrixXi* Fmesh_ptr
);

// Per-cell stepping (inner iteration)
void step_cell_inner_iteration(Cell& cell, const ContouringOptions& options);

// Execute assign_spheres_to_cells and compute_face_cells_intersections (outer iteration)
void start_outer_iteration(
    std::vector<Cell>& cells,
    const Eigen::VectorXd& S,
    const Eigen::MatrixXd& GV,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    int resX,
    int resY,
    int resZ,
    const ContouringOptions& opts
);

