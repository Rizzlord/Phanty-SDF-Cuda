#include "Cell.h"
#include <iostream>
#include <Eigen/Dense>
#include <polyscope/polyscope.h>
#include <polyscope/point_cloud.h>
#include <polyscope/curve_network.h>
#include <polyscope/surface_mesh.h>
#include <array>



std::pair<int, int> get_face_normals_from_edge_idx(int edge_idx){
    // 0:+X,1:-X,2:+Y,3:-Y,4:+Z,5:-Z

    // Mapping for the 12 edges -> pair of adjacent face normals
    static const std::array<std::pair<int,int>, 12> edge_to_face_normals = {{
        std::make_pair(3, 5), // edge 0: (0,1)
        std::make_pair(0, 5), // edge 1: (1,2)
        std::make_pair(2, 5), // edge 2: (2,3)
        std::make_pair(1, 5), // edge 3: (3,0)
        std::make_pair(3, 4), // edge 4: (4,5)
        std::make_pair(0, 4), // edge 5: (5,6)
        std::make_pair(2, 4), // edge 6: (6,7)
        std::make_pair(1, 4), // edge 7: (7,4)
        std::make_pair(1, 3), // edge 8: (0,4) vertical
        std::make_pair(0, 3), // edge 9: (1,5) vertical
        std::make_pair(0, 2), // edge 10: (2,6) vertical
        std::make_pair(1, 2)  // edge 11: (3,7) vertical
    }};
    return edge_to_face_normals[edge_idx];
}


// Linear interpolation factor t such that f(a + t(b-a)) = iso
inline double lerp_t(double fa, double fb, double iso = 0.0)
{
    return (fa - iso) / (fa - fb + 1e-16); // avoid division by zero
}

// ============================================================================
// Constructor
// ============================================================================
Cell::Cell(int i, int j, int k,
           Eigen::Matrix<double, 8, 3> corner_positions,
           Eigen::Matrix<double, 8, 1> corner_sdf_values,
           Eigen::Matrix<double, 8, 3> corner_sdf_normals)
    : ix(i), iy(j), iz(k),
      corners(corner_positions),
      cornerSDF(corner_sdf_values),
      cornerNormals(corner_sdf_normals)
{
    // Detect if this cell has a sign change (surface crosses this cell).
    // A zero-crossing exists if some corner has SDF < 0 and another has SDF > 0.
    bool has_neg = false;
    bool has_pos = false;

    for (int c = 0; c < 8; ++c) {
        if (cornerSDF(c) < 0) has_neg = true;
        if (cornerSDF(c) > 0) has_pos = true;
    }

    has_vertex = (has_neg && has_pos);
}


Eigen::Vector3d gradientAtPointUsingTrilinear(
    const Eigen::Matrix<double, 8, 1>& cornerSDF,
    const Eigen::Matrix<double, 8, 3>& corners,
    const Eigen::Vector3d& p)
{
    // Corner SDFs in canonical order:
    // 0: (0,0,0), 1: (1,0,0), 2: (1,1,0), 3: (0,1,0),
    // 4: (0,0,1), 5: (1,0,1), 6: (1,1,1), 7: (0,1,1)

    const double S000 = cornerSDF(0);
    const double S100 = cornerSDF(1);
    const double S110 = cornerSDF(2);
    const double S010 = cornerSDF(3);
    const double S001 = cornerSDF(4);
    const double S101 = cornerSDF(5);
    const double S111 = cornerSDF(6);
    const double S011 = cornerSDF(7);

    // Recover physical spacing from corners (axis-aligned assumption)
    const double x0 = corners(0, 0);
    const double x1 = corners(1, 0);
    const double y0 = corners(0, 1);
    const double y1 = corners(3, 1);
    const double z0 = corners(0, 2);
    const double z1 = corners(4, 2);

    const double dx = x1 - x0;
    const double dy = y1 - y0;
    const double dz = z1 - z0;

    const double eps = 1e-12;

    // Local parametric coords (u,v,w) in [0,1]^3
    double u = 0.5, v = 0.5, w = 0.5; // safe defaults if degenerate
    if (std::abs(dx) > eps) u = (p.x() - x0) / dx;
    if (std::abs(dy) > eps) v = (p.y() - y0) / dy;
    if (std::abs(dz) > eps) w = (p.z() - z0) / dz;

    // Clamp to [0,1] for numerical robustness
    auto clamp01 = [](double t) { return std::max(0.0, std::min(1.0, t)); };
    u = clamp01(u);
    v = clamp01(v);
    w = clamp01(w);

    const double omu = 1.0 - u;
    const double omv = 1.0 - v;
    const double omw = 1.0 - w;

    // Trilinear df/du, df/dv, df/dw in param space
    double df_du =
          (S100 - S000) * omv * omw
        + (S110 - S010) * v   * omw
        + (S101 - S001) * omv * w
        + (S111 - S011) * v   * w;

    double df_dv =
          (S010 - S000) * omu * omw
        + (S110 - S100) * u   * omw
        + (S011 - S001) * omu * w
        + (S111 - S101) * u   * w;

    double df_dw =
          (S001 - S000) * omu * omv
        + (S101 - S100) * u   * omv
        + (S011 - S010) * omu * v
        + (S111 - S110) * u   * v;

    Eigen::Vector3d g = Eigen::Vector3d::Zero();

    // Chain rule: x = x0 + u*dx, etc → df/dx = df/du * du/dx = df/du / dx
    if (std::abs(dx) > eps) g.x() = df_du / dx;
    if (std::abs(dy) > eps) g.y() = df_dv / dy;
    if (std::abs(dz) > eps) g.z() = df_dw / dz;

    return g;
}


// ============================================================================
// Fill Hermite data by checking every edge for a sign change and linearly
// interpolating both intersection position and normals.
// ============================================================================
void Cell::fill_hermite_data(
    const TrueSdfFunc* true_sdf,
    const TrueSdfGradFunc* true_sdf_grad)
{
    hermite_positions.clear();
    hermite_normals.clear();

    if (!has_vertex) {
        return;
    }

    const bool use_true = (true_sdf != nullptr && true_sdf_grad != nullptr);

    // Iterate over all 12 edges
    for (int i = 0; i < 12; ++i) {

        int a = EDGE_PAIRS[i][0];
        int b = EDGE_PAIRS[i][1];

        double fa = cornerSDF(a);
        double fb = cornerSDF(b);

        // No sign change → skip this edge
        if (fa * fb > 0.0) continue;

        Eigen::Vector3d pa = corners.row(a).transpose();
        Eigen::Vector3d pb = corners.row(b).transpose();

        const int edge_index = i; 

        if (!use_true) {
            // ============================
            // 
            // ============================
            double t = lerp_t(fa, fb);  // between 0 and 1

            Eigen::Vector3d p =
                (1.0 - t) * pa + t * pb;

            // Eigen::Vector3d n =
            //     (1.0 - t) * cornerNormals.row(a).transpose() +
            //        t      * cornerNormals.row(b).transpose();

            Eigen::Vector3d n = gradientAtPointUsingTrilinear(cornerSDF, corners, p);

            if (n.norm() > 0.0) {
                n.normalize();
            }

            hermite_positions[edge_index] = p;
            hermite_normals[edge_index] = n;
        } else {
            // ============================
            // TRUE SDF / TRUE GRAD PATH
            // ============================
            Eigen::Vector3d p0 = pa;
            Eigen::Vector3d p1 = pb;

            double f0 = (*true_sdf)(p0);
            double f1 = (*true_sdf)(p1);

            // If even the true SDF doesn’t see a sign change, fall back
            if (f0 * f1 > 0.0) {
            // if (true) {zß
                double t = lerp_t(fa, fb);

                Eigen::Vector3d p =
                    (1.0 - t) * pa + t * pb;

                Eigen::Vector3d n = gradientAtPointUsingTrilinear(cornerSDF, corners, p);

                if (n.norm() > 0.0) n.normalize();

                hermite_positions[edge_index] = p;
                hermite_normals[edge_index] = n;
                continue;
            }

            // Binary search along the edge for the zero crossing of true_sdf
            const int maxIter = 16;
            const double tol = 1e-6;

            for (int iter = 0; iter < maxIter; ++iter) {
                Eigen::Vector3d pm = 0.5 * (p0 + p1);
                double fm = (*true_sdf)(pm);

                // Stop if small enough
                if (std::abs(fm) < tol) {
                    p0 = p1 = pm;
                    break;
                }

                // Maintain bracket [p0, p1] with sign change
                if (f0 * fm <= 0.0) {
                    p1 = pm;
                    f1 = fm;
                } else {
                    p0 = pm;
                    f0 = fm;
                }

                if ((p1 - p0).norm() < tol) {
                    break;
                }
            }

            Eigen::Vector3d p = 0.5 * (p0 + p1);
            Eigen::Vector3d n = (*true_sdf_grad)(p);

            if (n.norm() > 0.0) {
                n.normalize();
            }

            hermite_positions[edge_index] = p;
            hermite_normals[edge_index] = n;
        }
    }
}

// ============================================================
// update(): QEF minimization placeholder
// ============================================================
void Cell::update(const double regularization_weight)
{
    if (!has_vertex || hermite_positions.empty()) {
        vertex = Eigen::Vector3d::Zero();
        has_vertex = false;
        return;
    }

    const int m = static_cast<int>(hermite_positions.size());
    Eigen::MatrixXd A(m, 3);
    Eigen::VectorXd b(m);

    // Build A and b
    int i = 0;
    for (const auto& pair : hermite_positions) {
        int edge_index = pair.first;
        const Eigen::Vector3d& p = pair.second;
        
        // Get normal using the edge_index key from the hermite_normals map
        Eigen::Vector3d n = hermite_normals.at(edge_index); 

        if (n.norm() > 0) {
            n.normalize(); // Normalize the copy
        }

        A.row(i) = n.transpose();
        b(i)     = n.dot(p);
        i++;
    }

    vertex = solve_quadratic_system(A, b, 0.01);

    // debug overload
    // vertex = centroid;

    has_vertex = true;
}



Eigen::Vector3d Cell::solve_quadratic_system(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    double svd_threshold
) {
    Eigen::Matrix3d ATA = A.transpose() * A;
    Eigen::Vector3d ATb = A.transpose() * b;

    // Compute SVD of AtA
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        ATA, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Vector3d S = svd.singularValues();
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    Eigen::Vector3d S_inv;
    for (int i = 0; i < 3; ++i) {
        if (S(i) > svd_threshold)
            S_inv(i) = 1.0 / S(i);
        else
            S_inv(i) = 0.0;                // truncate small singular values
    }

    //std::cout << "Singular values: " << S_inv.transpose() << std::endl;
    // Pseudoinverse of ATA:
    // (ATA)^+ = V * diag(S_inv) * U^T
    Eigen::Matrix3d ATA_pinv = V * S_inv.asDiagonal() * U.transpose();
    Eigen::Vector3d centroid = getCentroid();

    // “Right-hand side” centered at centroid:
    // c = (ATA)^+ (ATb - ATA * centroid)
    Eigen::Vector3d rhs = ATb - ATA * centroid;
    Eigen::Vector3d c   = ATA_pinv * rhs;

    // Final solution:
    return centroid + c;
}


void Cell::minimize_qef(
    double mu,
    double dc_weight,
    double sphere_weight,
    double svd_threshold,
    bool verbose
) {
    // Build positions and normals in a consistent order by iterating over
    // hermite_positions and looking up the corresponding normal by the same key.
    std::vector<Eigen::Vector3d> positions;
    std::vector<Eigen::Vector3d> normals;
    positions.reserve(hermite_positions.size());
    normals.reserve(hermite_positions.size());

    for (const auto& kv : hermite_positions) {
        int edge_index = kv.first;
        positions.push_back(kv.second);

        // Find matching normal. If missing, push a zero vector as a fallback.
        auto nit = hermite_normals.find(edge_index);
        if (nit != hermite_normals.end()) {
            Eigen::Vector3d n = nit->second;
            normals.push_back(n);
        } else {
            normals.push_back(Eigen::Vector3d::Zero());
        }
    }

    int num_normals = (int) normals.size();
    int num_spheres = (int) closest_points_info.size();
    int num_eqs_regularization = 3;
    Eigen::MatrixXd A(num_normals + num_spheres + num_eqs_regularization, 3);
    Eigen::VectorXd b(num_normals + num_spheres + num_eqs_regularization);

    // Initialize A and b to zero
    A.setZero();
    b.setZero();

    int current_row = 0;
    for (size_t i = 0; i < num_normals; ++i) {
        Eigen::Vector3d scaled_normal = dc_weight * normals[i];
        A.row(current_row) = scaled_normal.transpose();
        b(current_row) = scaled_normal.dot(positions[i]);
        current_row++;
    }

    const double sqrt_sphere_weight = std::sqrt(sphere_weight);
    //std::cout << "Number of sphere constraints: " << num_spheres << std::endl;
    for (const auto& info : closest_points_info) {
        double alpha = info.barycentric_coords(0);
        double beta  = info.barycentric_coords(1);
        double gamma = info.barycentric_coords(2);

        if (abs(alpha) < 1e-6){
            // alpha = 0.001;
            alpha = 1.0;
            beta = 0.0;
            gamma = 0.0;
        }

        Eigen::Vector3d t = alpha * prev_vertex + beta * info.p + gamma * info.fip;
        Eigen::Vector3d c_to_t_mesh = t - info.c;
        double rho = c_to_t_mesh.norm();

        // radius is the distance between q and c
        double radius = (info.q - info.c).norm();

        if (rho > radius) {
            // Get the index of the sphere in cell.assigned_spheres
            int sphere_idx = info.sphere_idx;
            // Check the sign of the sphere
            int sign = assigned_spheres[sphere_idx].sign;
            if (sign == -1){
                continue;
            }
        }

        // Calculate q: the point on the sphere surface closest to t_mesh
        Eigen::Vector3d q_sphere;
        if (rho < 1e-9) {
            q_sphere = info.c; 
        } else {
            q_sphere = info.c + (c_to_t_mesh / rho) * radius;
        }
        
        Eigen::Vector3d q = q_sphere;
        Eigen::Vector3d d = q - info.c;
        Eigen::Vector3d p = info.p;
        Eigen::Vector3d fip = info.fip;

        A.row(current_row) = alpha * sqrt_sphere_weight * d.transpose();
        double b_row_not_scaled = q.dot(d) 
                                 - beta * p.dot(d)
                                 - gamma * fip.dot(d);
        b(current_row) = sqrt_sphere_weight * b_row_not_scaled;
        current_row++;
    }

    double sqrt_mu = std::sqrt(mu);
    for (int i = 0; i < 3; ++i) {
        A.row(current_row) = sqrt_mu * Eigen::Vector3d::Unit(i).transpose();
        b(current_row) = sqrt_mu * prev_vertex(i);
        current_row++;
    }   

    vertex = solve_quadratic_system(A, b, svd_threshold);

    Eigen::VectorXd residual = A * vertex - b;
    energy = residual.squaredNorm();
}


// void Cell::minimize_qef(
//     double mu,
//     double dc_weight,
//     double sphere_weight,
//     double svd_threshold,
//     bool verbose
// ) {
//     // Build positions and normals in a consistent order by iterating over
//     // hermite_positions and looking up the corresponding normal by the same key.
//     std::vector<Eigen::Vector3d> positions;
//     std::vector<Eigen::Vector3d> normals;
//     positions.reserve(hermite_positions.size());
//     normals.reserve(hermite_positions.size());

//     for (const auto& kv : hermite_positions) {
//         int edge_index = kv.first;
//         positions.push_back(kv.second);

//         // Find matching normal. If missing, push a zero vector as a fallback.
//         auto nit = hermite_normals.find(edge_index);
//         if (nit != hermite_normals.end()) {
//             Eigen::Vector3d n = nit->second;
//             normals.push_back(n);
//         } else {
//             normals.push_back(Eigen::Vector3d::Zero());
//         }
//     }

//     int num_normals = (int) normals.size();
//     int num_spheres = (int) closest_points_info.size();
//     int num_eqs_regularization = 3;
//     Eigen::MatrixXd A(num_spheres * 3, 3);
//     Eigen::VectorXd b(num_spheres * 3);

//     int current_row = 0;
//     //std::cout << "Number of normals: " << num_normals << std::endl;
//     // for (size_t i = 0; i < num_normals; ++i) {
//     //     Eigen::Vector3d scaled_normal = dc_weight * normals[i];
//     //     A.row(current_row) = scaled_normal.transpose();
//     //     b(current_row) = scaled_normal.dot(positions[i]);
//     //     current_row++;
//     // }

//     const double sqrt_sphere_weight = std::sqrt(sphere_weight);
//     //std::cout << "Number of sphere constraints: " << num_spheres << std::endl;
//     for (const auto& info : closest_points_info) {
//         double alpha = info.barycentric_coords(0);
//         double beta  = info.barycentric_coords(1);
//         double gamma = info.barycentric_coords(2);

//         if (abs(alpha) < 1e-6){
//             // alpha = 0.001;
//             alpha = 1.0;
//             beta = 0.0;
//             gamma = 0.0;
//         }

//         Eigen::Vector3d t = alpha * prev_vertex + beta * info.p + gamma * info.fip;
//         Eigen::Vector3d c_to_t_mesh = t - info.c;
//         double rho = c_to_t_mesh.norm();

//         // Calculate q: the point on the sphere surface closest to t_mesh
//         Eigen::Vector3d q_sphere;
//         if (rho < 1e-9) {
//             q_sphere = info.c; 
//         } else {
//             // radius is the distance between q and c
//             double radius = (info.q - info.c).norm();
//             q_sphere = info.c + (c_to_t_mesh / rho) * radius;
//         }

        
//         Eigen::Vector3d q = q_sphere;
//         Eigen::Vector3d d = q - info.c;
//         Eigen::Vector3d p = info.p;
//         Eigen::Vector3d fip = info.fip;

//         // A.row(current_row) = alpha * sqrt_sphere_weight * d.transpose();
//         // double b_row_not_scaled = q.dot(d) 
//         //                          - beta * p.dot(d)
//         //                          - gamma * fip.dot(d);
//         // b(current_row) = sqrt_sphere_weight * b_row_not_scaled;
//         // current_row++;

//         A.block<3,3>(current_row, 0) =
//             alpha * sqrt_sphere_weight * Eigen::Matrix3d::Identity();

//         b.segment<3>(current_row) =
//             sqrt_sphere_weight * (q - beta * p - gamma * fip);
//         current_row += 3;
//     }

//     // double sqrt_mu = std::sqrt(mu);
//     // for (int i = 0; i < 3; ++i) {
//     //     A.row(current_row) = sqrt_mu * Eigen::Vector3d::Unit(i).transpose();
//     //     b(current_row) = sqrt_mu * prev_vertex(i);
//     //     current_row++;
//     // }   

//     //Print A and b
//     // std::cout << "Matrix A\n" << A << std::endl << std::endl;
//     // std::cout << "Vector b" << b << std::endl  << std::endl << std::endl;

//     vertex = solve_quadratic_system(A, b, svd_threshold);

//     Eigen::VectorXd residual = A * vertex - b;
//     energy = residual.squaredNorm();
// }


Eigen::Vector3d Cell::getCentroid(){
    // centroid of hermite positions
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    if(hermite_positions.empty()){
        return centroid;
    }
    
    // Iterate over the values (Vector3d) stored in the map
    for(const auto& pair : hermite_positions){
        centroid += pair.second;
    }
    
    centroid /= double(hermite_positions.size());
    return centroid;
}

void Cell::clean() {
    closest_points_info.clear();
    assigned_spheres.clear();
    closest_faces.clear();
    local_mesh_vertices.clear();
    local_mesh_faces.clear();
    inner_iter = 0;
}



#include <glm/glm.hpp>

void Cell::visualize() const {
    // Register structures with a unique prefix so we don't stomp on global scene objects.
    std::string prefix = "Cell_" + std::to_string(ix) + "_" + std::to_string(iy) + "_" + std::to_string(iz) + "_";

    // Remove any previous per-cell structures with the same names (no error if absent)
    auto tryRemove = [&](const std::string& name) {
        polyscope::removeStructure(prefix + name, /*errorIfAbsent=*/false);
    };

    // -----------------------------------
    // 1. Show Hermite points (gray)
    // -----------------------------------
    if (!hermite_positions.empty()) {
        Eigen::MatrixXd H(hermite_positions.size(), 3);
        int i = 0;
        for (const auto& pair : hermite_positions) {
            H.row(i) = pair.second.transpose();
            i++;
        }
        tryRemove("Hermite points");
        auto* pc_herm = polyscope::registerPointCloud(prefix + "Hermite points", H);
        pc_herm->setPointColor(glm::vec3(0.5, 0.5, 0.5));  // GRAY
        // Per-point color quantity so we can change individual Hermite point colors (e.g. on click)
        std::vector<glm::vec3> hermiteColors((size_t)hermite_positions.size(), glm::vec3(0.5f, 0.5f, 0.5f));
        // Add as a color quantity; name chosen to be unique-ish and descriptive
        pc_herm->addColorQuantity("Hermite color", hermiteColors)->setEnabled(true);
    }

    // -----------------------------------
    // 2. Hermite normals (orange)
    // -----------------------------------
    if (!hermite_positions.empty() && hermite_normals.size() == hermite_positions.size()) {
        Eigen::MatrixXd N(hermite_normals.size(), 3);
        int i = 0;
        // We must iterate over hermite_positions to ensure the points and normals 
        // matrices H and N align based on the order of iteration.
        for (const auto& pair : hermite_positions) { 
            const int edge_index = pair.first;
            // Access normal using the edge_index key
            N.row(i) = hermite_normals.at(edge_index).transpose();
            i++;
        }

        polyscope::getPointCloud(prefix + "Hermite points")
            ->addVectorQuantity("Hermite normals", N)
            ->setVectorColor(glm::vec3(1.0, 0.5, 0.0)); // ORANGE
    }

    // -----------------------------------
    // 3. Cell corners (green)
    // -----------------------------------
    Eigen::MatrixXd C(8, 3);
    C = corners;
    tryRemove("Corners");
    auto* pc_corners = polyscope::registerPointCloud(prefix + "Corners", C);
    pc_corners->setPointColor(glm::vec3(0.0, 1.0, 0.0)); // GREEN

    // -----------------------------------
    // 4. Corner SDF and normals
    // -----------------------------------
    {
        Eigen::VectorXd sdfVals = cornerSDF;
        pc_corners->addScalarQuantity("corner SDF", sdfVals)->setMapRange({-1.0, 1.0});

        pc_corners->addVectorQuantity("corner normals", cornerNormals)
            ->setVectorColor(glm::vec3(1.0, 0.5, 0.0)); // ORANGE

    }

    // -----------------------------------
    // 5. Cell vertex (blue)
    // -----------------------------------
    {
        Eigen::MatrixXd Vtx(1, 3);
        Vtx.row(0) = vertex.transpose();
    tryRemove("Cell vertex");
    auto* pc_v = polyscope::registerPointCloud(prefix + "Cell vertex", Vtx);
        pc_v->setPointColor(glm::vec3(0.1, 0.1, 1.0));   // BLUE
        // Larger point size
        pc_v->setPointRadius(0.02);
    }

    // -----------------------------------
    // 6. Cell edges (black)
    // -----------------------------------
    {
        // 12 edges of the hexahedron
        std::vector<std::pair<int,int>> edges = {
            {0,1},{1,2},{2,3},{3,0},        // bottom face
            {4,5},{5,6},{6,7},{7,4},        // top face
            {0,4},{1,5},{2,6},{3,7}         // vertical connections
        };

        Eigen::MatrixXi E(edges.size(), 2);
        for (int e = 0; e < edges.size(); ++e) {
            E(e,0) = edges[e].first;
            E(e,1) = edges[e].second;
        }

        tryRemove("Cell edges");
        auto* cn = polyscope::registerCurveNetwork(prefix + "Cell edges", C, E);
        cn->setColor(glm::vec3(0.0, 0.0, 0.0)); // BLACK
        cn->setRadius(0.003);
    }

    // -----------------------------------
    // 7. Assigned spheres
    // -----------------------------------
    if (!assigned_spheres.empty()) {
        Eigen::MatrixXd P(assigned_spheres.size(), 3);
        Eigen::VectorXd R(assigned_spheres.size());

        Eigen::MatrixXd sphereColors(assigned_spheres.size(), 3);
        Eigen::MatrixXd centerColors(assigned_spheres.size(), 3);

        for (int i = 0; i < (int)assigned_spheres.size(); ++i) {
            P.row(i) = assigned_spheres[i].center.transpose();
            R(i)     = assigned_spheres[i].radius;

            // Give color based on sign
            if (assigned_spheres[i].sign < 0) {
                // Positive sphere - light blue
                sphereColors.row(i) = Eigen::Vector3d(0.2, 0.6, 1.0);
                centerColors.row(i) = Eigen::Vector3d(0.0, 0.0, 1.0);
            } else {
                // Negative sphere - light red
                sphereColors.row(i) = Eigen::Vector3d(1.0, 0.3, 0.3);
                centerColors.row(i) = Eigen::Vector3d(1.0, 0.0, 0.0);
            }
        }

        // Spheres as semi-transparent
    tryRemove("Assigned Spheres");
    auto* pc_spheres = polyscope::registerPointCloud(prefix + "Assigned Spheres", P);
        pc_spheres->addScalarQuantity("radius", R)->setEnabled(true);
        pc_spheres->setPointRadiusQuantity("radius", false);
        pc_spheres->setPointRenderMode(polyscope::PointRenderMode::Sphere);

        pc_spheres->setTransparency(0.3);  // 30% opacity

        // Mark centers explicitly as small orange points
    tryRemove("Sphere Centers");
    auto* pc_centers = polyscope::registerPointCloud(prefix + "Sphere Centers", P);
    pc_centers->setPointColor(glm::vec3(1.0, 0.5, 0.0)); // orange
    pc_centers->setPointRadius(0.01);                     // small dot
    pc_centers->setPointRenderMode(polyscope::PointRenderMode::Sphere);

        // Set per-sphere colors
        pc_spheres->addColorQuantity("Sign Color", sphereColors)->setEnabled(true);
        pc_centers->addColorQuantity("Center Color", centerColors)->setEnabled(true);
    }


    // -----------------------------------
    // 8. Closest Points Info (Magenta/Cyan)
    // -----------------------------------
    // if (!closest_points_info.empty()) {

    //     std::vector<Eigen::Vector3d> pointsQ;
    //     std::vector<Eigen::Vector3d> pointsT;
    //     std::vector<Eigen::Vector3d> segmentNodes;
    //     std::vector<std::array<int, 2>> segmentEdges;
    //     std::vector<std::array<int, 2>> connectionEdges;

    //     pointsQ.reserve(closest_points_info.size());
    //     pointsT.reserve(closest_points_info.size());
        
    //     int nodeIdx = 0;

    //     for (const auto& info : closest_points_info) {
    //         // Recompute t (projection on segment) for visualization
    //         // t = p + lambda * (v - p)
    //         Eigen::Vector3d v_minus_p = vertex - info.p;
    //         Eigen::Vector3d t = info.p + info.lambda * v_minus_p;

    //         pointsQ.push_back(info.q);
    //         pointsT.push_back(t);

            
    //         segmentNodes.push_back(t);
    //         segmentNodes.push_back(info.c);
    //         connectionEdges.push_back({nodeIdx, nodeIdx + 1});
    //         nodeIdx += 2;
    //     }

    //     // Visualize Q points (Closest on Sphere Surface) - Cyan
    //     if(!pointsQ.empty()) {
    //         Eigen::MatrixXd matQ(pointsQ.size(), 3);
    //         for(size_t i=0; i<pointsQ.size(); ++i) matQ.row(i) = pointsQ[i].transpose();
    //         auto* pc_q = polyscope::registerPointCloud(prefix + "Closest Points Q", matQ);
    //         pc_q->setPointColor(glm::vec3(0.0, 1.0, 1.0)); // Cyan
    //         pc_q->setPointRadius(0.012);
    //     }

    //     // Visualize T points (Projection on Spoke) - Magenta
    //     if(!pointsT.empty()) {
    //         Eigen::MatrixXd matT(pointsT.size(), 3);
    //         for(size_t i=0; i<pointsT.size(); ++i) matT.row(i) = pointsT[i].transpose();
    //         auto* pc_t = polyscope::registerPointCloud(prefix + "Projection Points T", matT);
    //         pc_t->setPointColor(glm::vec3(1.0, 0.0, 1.0)); // Magenta
    //         pc_t->setPointRadius(0.012);
    //     }

    //     // Visualize Connection Lines (T -> Center) - White/Grey
    //     if(!segmentNodes.empty()) {
    //         Eigen::MatrixXd matNodes(segmentNodes.size(), 3);
    //         Eigen::MatrixXi matEdges(connectionEdges.size(), 2);
            
    //         for(size_t i=0; i<segmentNodes.size(); ++i) matNodes.row(i) = segmentNodes[i].transpose();
    //         for(size_t i=0; i<connectionEdges.size(); ++i) {
    //             matEdges(i, 0) = connectionEdges[i][0];
    //             matEdges(i, 1) = connectionEdges[i][1];
    //         }

    //         auto* cn_conn = polyscope::registerCurveNetwork(prefix + "Sphere-Spoke Connections", matNodes, matEdges);
    //         cn_conn->setColor(glm::vec3(0.8, 0.8, 0.8));
    //         cn_conn->setRadius(0.002);
    //     }
    // }

    // -----------------------------------
    // 9. Face Intersections (brown)
    // -----------------------------------
    if (!face_intersections.empty()) {
        std::vector<Eigen::Vector3d> pointsInter;
        pointsInter.reserve(face_intersections.size());
        for (const auto& fi : face_intersections) {
            pointsInter.push_back(fi.second);
        }
        Eigen::MatrixXd matInter(pointsInter.size(), 3);
        for (size_t i = 0; i < pointsInter.size(); ++i) {
            matInter.row(i) = pointsInter[i].transpose();
        }
        tryRemove("Face Intersections");
        auto* pc_inter = polyscope::registerPointCloud(prefix + "Face Intersections", matInter);
        pc_inter->setPointColor(glm::vec3(0.5, 0.3, 0.2)); // Brown
        pc_inter->setPointRadius(0.025);
    }    


    // -----------------------------------
    // 10. Cell Triangulation (if computed)
    // -----------------------------------
    if (!local_mesh_vertices.empty() && !local_mesh_faces.empty()) {
        Eigen::MatrixXd V_mesh(local_mesh_vertices.size(), 3);
        for (size_t i = 0; i < local_mesh_vertices.size(); ++i) {
            V_mesh.row(i) = local_mesh_vertices[i].transpose();
        }

        Eigen::MatrixXi F_mesh(local_mesh_faces.size(), 3);
        for (size_t i = 0; i < local_mesh_faces.size(); ++i) {
            F_mesh.row(i) = local_mesh_faces[i];
        }

        if (F_mesh.rows() > 0) {
            tryRemove("Cell Mesh");
            polyscope::registerSurfaceMesh(prefix + "Cell Mesh", V_mesh, F_mesh)
                ->setEdgeColor(glm::vec3(0.0, 0.5, 0.5)) // Teal edges
                ->setSurfaceColor(glm::vec3(0.8, 0.8, 0.9)) // Light Blue face
                ->setEdgeWidth(1.0);
        }
    }

    // -----------------------------------
    // 11. Closest points info 
    // -----------------------------------
    // Plot the t points stored in closest_points_info
    // Reconstruct the t using the stored barycentric coordinates (magenta)
    // Also plot the corresponding q points (cyan)
    
    if (!closest_points_info.empty()) {
        std::vector<Eigen::Vector3d> pointsQ;
        std::vector<Eigen::Vector3d> pointsT;
        pointsQ.reserve(closest_points_info.size());
        pointsT.reserve(closest_points_info.size());

        for (const auto& info : closest_points_info) {

            // Reconstruct t using barycentric coordinates
            Eigen::Vector3d t = info.barycentric_coords(0) * prev_vertex +
                                info.barycentric_coords(1) * info.p +
                                info.barycentric_coords(2) * info.fip;
            pointsT.push_back(t);
            pointsQ.push_back(info.q);
        }
        Eigen::MatrixXd matQ(pointsQ.size(), 3);
        for (size_t i = 0; i < pointsQ.size(); ++i) {
            matQ.row(i) = pointsQ[i].transpose();
        }
        tryRemove("Q Points");
        auto* pc_q = polyscope::registerPointCloud(prefix + "Q Points", matQ);
        pc_q->setPointColor(glm::vec3(0.0, 1.0, 1.0)); // Cyan
        pc_q->setPointRadius(0.012);

        Eigen::MatrixXd matT(pointsT.size(), 3);
        for (size_t i = 0; i < pointsT.size(); ++i) {
            matT.row(i) = pointsT[i].transpose();
        }
        tryRemove("T Points");
        auto* pc_t = polyscope::registerPointCloud(prefix + "T Points", matT);
        pc_t->setPointColor(glm::vec3(1.0, 0.0, 1.0)); // Magenta
        pc_t->setPointRadius(0.012);
    }


    // No blocking show() here — we register structures into the existing Polyscope context and return.
}