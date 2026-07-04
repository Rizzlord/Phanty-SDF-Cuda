#include "contouring.h"

#include <igl/marching_cubes.h>
#include <iostream>

void contouring(
    const Eigen::VectorXd& S,
    const Eigen::MatrixXd& GV,
    int resX,
    int resY,
    int resZ,
    double isoValue,
    Eigen::MatrixXd& V,
    Eigen::MatrixXi& F,
    const ContouringOptions& options
)
{
    if (options.verbose) {
        std::cout << "[contouring] method = "
                  << (options.method == ContouringMethod::DualContouring ?
                      "DualContouring" : "MarchingCubes")
                  << ", grid = " << resX << " x " << resY << " x " << resZ
                  << ", iso = " << isoValue << std::endl;
    }

    if (options.method == ContouringMethod::MarchingCubes) {

        // Directly call libigl marching cubes
        igl::marching_cubes(S, GV, resX, resY, resZ, isoValue, V, F);
        return;
    }

    // ===========================
    // Dual Contouring placeholder
    // ===========================
    if (options.method == ContouringMethod::DualContouring) {

        std::cerr << "[contouring] WARNING: "
                  << "Dual contouring not implemented yet. "
                  << "Falling back to marching cubes.\n";

        igl::marching_cubes(S, GV, resX, resY, resZ, isoValue, V, F);
        return;
    }
}