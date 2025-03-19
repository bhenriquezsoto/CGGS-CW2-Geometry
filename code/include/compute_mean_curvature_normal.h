#ifndef COMPUTE_MEAN_CURVATURE_NORMAL_HEADER_FILE
#define COMPUTE_MEAN_CURVATURE_NORMAL_HEADER_FILE

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <cmath>

void compute_mean_curvature_normal(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::SparseMatrix<double>& L,
    const Eigen::VectorXd& vorAreas,
    Eigen::MatrixXd& Hn,
    Eigen::VectorXd& H
) {
    using Vec3 = Eigen::Vector3d;
    const int nV = V.rows(), nF = F.rows();

    // 1) Mean-curvature normal = (L * V) / (2*area)
    // Compute L * V first
    Hn = Eigen::MatrixXd::Zero(nV, 3);
    Eigen::MatrixXd LV = L * V;

    // Divide by 2 * Voronoi area for each vertex
    for(int i = 0; i < nV; ++i) {
        // Avoid division by very small areas
        double area = std::max(vorAreas(i), 1e-12);
        Hn.row(i) = LV.row(i) / (2.0 * area);
    }

    // 2) Compute vertex normals by averaging face normals
    std::vector<std::vector<int>> vertex_to_faces(nV);
    find_adjacent_faces(V, F, nV, vertex_to_faces);
    Eigen::MatrixXd vertexNormals = Eigen::MatrixXd::Zero(nV, 3);
    
    for(int i = 0; i < nV; ++i) {
        Vec3 sum_normals = Vec3::Zero();
        
        // Use pre-computed adjacent faces
        for(int f : vertex_to_faces[i]) {
            // Calculate normal of face
            Vec3 v0 = V.row(F(f,0));
            Vec3 v1 = V.row(F(f,1));
            Vec3 v2 = V.row(F(f,2));
            Vec3 normal = (v1 - v0).cross(v2 - v0).normalized();
            sum_normals += normal;
        }
        vertexNormals.row(i) = sum_normals.normalized();
    }

    // 3) Signed mean curvature = sign(Hn·n) * |Hn|
    H = Eigen::VectorXd(nV);
    for(int i = 0; i < nV; ++i) {
        double dot_product = Hn.row(i).dot(vertexNormals.row(i));
        double magnitude = Hn.row(i).norm();
        H(i) = (dot_product >= 0) ? magnitude : -magnitude;
    }
}

#endif
