#pragma once

#include <Eigen/Core>
#include <functional>

// A SDF is a function taking a 3D row-vector point and returning a signed distance.
using SDF = std::function<double(const Eigen::RowVector3d&)>;

// A gradient function returns a normalized gradient (normal) at the point.
using SDFGrad = std::function<Eigen::RowVector3d(const Eigen::RowVector3d&)>;

// Compute a finite-difference gradient for any SDF using centered differences.
// h is the finite-difference step (small positive number).
SDFGrad finite_difference_gradient(const SDF& f, double h);

// Factory: rotated axis-aligned box SDF (half-extents b, rotation matrix R)
// Returns pair {f, f_grad_fd}. f_grad_fd is a normalized finite-difference gradient
// using a small step derived from provided h (if h <= 0, default small step used).
struct SDFObject {
    SDF f;
    SDFGrad grad;
};


SDFObject make_rotated_box_sdf(const Eigen::RowVector3d& half_extents,
                                const Eigen::Matrix3d& R,
                                double fd_h = -1.0);

SDFObject make_rotated_box_sdf(
    const Eigen::RowVector3d& b,
    const Eigen::RowVector3d& center, 
    const Eigen::Matrix3d& R,
    double fd_h = -1.0
);


SDFObject make_torus_sdf(
    double ra, 
    double rb,
    double fd_h
);

SDFObject make_cylinder_sdf(
    double he, 
    double r,
    double fd_h
);

SDFObject make_triangular_prism_sdf(
    const Eigen::RowVector2d& half_extents, 
    const Eigen::Matrix3d& R,
    double fd_h);

SDFObject make_cut_sphere_sdf(double r, double h, double fd_h);

SDFObject make_octahedron_sdf(double s, const Eigen::Matrix3d& R, double fd_h);