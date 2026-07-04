#pragma once

#include <Eigen/Core>
#include <functional>
#include "Cell.h"

struct OctreeNode {
    int i, j, k;     // Grid indices of the node's origin (min corner)
    int size;        // Cell width in grid units (e.g. 1 for leaf nodes, 2 for their parents, etc.)
    int depth;

    bool is_leaf;
    std::array<std::unique_ptr<OctreeNode>, 8> children;

    // Leaf-only
    std::unique_ptr<Cell> cell;

    // Internal-node data (for simplification)
    Eigen::Vector3d vertex;
    int vertex_index = -1;
    bool has_vertex;

    Eigen::Matrix<double, 8, 1> cornerSDF;
    Eigen::Matrix<double, 8, 3> cornerNormals;
    
    OctreeNode() : i(0), j(0), k(0), size(0), depth(0), is_leaf(false), has_vertex(false) {}
    OctreeNode(int i, int j, int k, int s, int d) : i(i), j(j), k(k), size(s), depth(d) {}
};

class OctreeManager {
public:
    OctreeManager(int rootSize);
    //~OctreeManager();

    void build(const Eigen::VectorXd& S, const Eigen::MatrixXd& GV,int resX, int resY, int resZ);
    void simplify(double tol);
    void extractMesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F);

    const OctreeNode* getRoot() const { return root.get(); }
private:
    std::unique_ptr<OctreeNode> root;
    // Recursive helpers
    void buildRecursive(OctreeNode* node, const Eigen::VectorXd& S, 
        const Eigen::MatrixXd& GV, int resX, int resY, int resZ);

    void processEdges(OctreeNode* node, int axis);
    void processEdge(OctreeNode* nodes[4], int axis);
    bool edgeHasSignChange(const OctreeNode* a,
                           const OctreeNode* b,
                           int axis) const;
};