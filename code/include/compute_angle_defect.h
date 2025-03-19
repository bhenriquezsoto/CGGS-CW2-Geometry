#ifndef COMPUTE_ANGLE_DEFECT_HEADER_FILE
#define COMPUTE_ANGLE_DEFECT_HEADER_FILE

#include <Eigen/Dense>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double angle_between_vectors(const Eigen::RowVector3d& v1, const Eigen::RowVector3d& v2) {
    double dot = v1.dot(v2);
    double norms = v1.norm() * v2.norm();
    // Avoid numerical issues with acos
    dot = std::max(-1.0, std::min(1.0, dot / norms));
    return std::acos(dot);
}

void find_adjacent_faces(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, int nV, std::vector<std::vector<int>>& vertex_to_faces) {
    for(int f = 0; f < F.rows(); ++f) {
        for(int j = 0; j < 3; ++j) {
            vertex_to_faces[F(f,j)].push_back(f);
        }
    }
}

Eigen::VectorXd compute_angle_defect(const Eigen::MatrixXd& V,
                                   const Eigen::MatrixXi& F,
                                   const Eigen::VectorXi& boundVMask) {
    const int nV = V.rows();
    Eigen::VectorXd G = Eigen::VectorXd::Zero(nV);
    
    // Pre-compute vertex to face mapping
    std::vector<std::vector<int>> vertex_to_faces(nV);
    find_adjacent_faces(V, F, nV, vertex_to_faces);
    
    for(int i = 0; i < nV; ++i) {
        double sum_angles = 0;
        
        // Use pre-computed adjacent faces
        for(int f : vertex_to_faces[i]) {
            // Find vertex position in face
            int j = 0;
            while(F(f,j) != i) ++j;
            
            // Calculate vectors from vertex to other vertices
            Eigen::RowVector3d v1 = V.row(F(f,(j+1)%3)) - V.row(i);
            Eigen::RowVector3d v2 = V.row(F(f,(j+2)%3)) - V.row(i);
            
            sum_angles += angle_between_vectors(v1, v2);
        }
        
        G[i] = (boundVMask[i] == 1) ? (M_PI - sum_angles) : (2 * M_PI - sum_angles);
    }
    
    return G;
}

#endif
