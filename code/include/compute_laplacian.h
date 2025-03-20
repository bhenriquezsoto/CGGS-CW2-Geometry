#ifndef COMPUTE_LAPLACIAN_HEADER_FILE
#define COMPUTE_LAPLACIAN_HEADER_FILE

#include "compute_angle_defect.h"
#include <Eigen/Dense>

void compute_laplacian(const Eigen::MatrixXd& V,
                       const Eigen::MatrixXi& F,
                       const Eigen::MatrixXi& E,
                       const Eigen::MatrixXi& EF,
                       const Eigen::VectorXi& boundEMask,
                       Eigen::SparseMatrix<double>& d0,
                       Eigen::SparseMatrix<double>& W,
                       Eigen::VectorXd& vorAreas){
    
    using namespace Eigen;
    using namespace std;
    d0.resize(E.rows(), V.rows());
    W.resize(E.rows(), E.rows());
    vorAreas = VectorXd::Zero(V.rows());
    std::vector<Triplet<double>> triplets;
    int nV = V.rows();
    std::vector<std::vector<int>> vertex_to_faces(nV);
    find_adjacent_faces(V, F, nV, vertex_to_faces);
    
    Vector3d v1, v2, v3;
    for(int i = 0; i < V.rows(); i++) {
        
        for (int j : vertex_to_faces[i]){
            v1 = V.row(F(j,0));
            v2 = V.row(F(j,1));
            v3 = V.row(F(j,2));
            
            double area = 0.5 * ((v2-v1).cross(v3-v1)).norm();
            vorAreas[i] += area/3;
        }
    }

    // Build edge-vertex incidence matrix
    for (int i = 0; i < E.rows(); i++) {
        triplets.push_back(Triplet<double>(i, E(i,0), -1));
        triplets.push_back(Triplet<double>(i, E(i,1), 1));
    }
    d0.setFromTriplets(triplets.begin(), triplets.end());
    
    // Check if it is a boundary edge
    for (int i = 0; i < E.rows(); i++) {
        int idface1 = EF(i,0);
        int opposite_vertex = EF(i,1);
        
        // Calculate vectors for the angle
        Vector3d edge1 = V.row(F(idface1,(opposite_vertex+1)%3)) - V.row(F(idface1,opposite_vertex));
        Vector3d edge2 = V.row(F(idface1,(opposite_vertex+2)%3)) - V.row(F(idface1,opposite_vertex));
        double angle = angle_between_vectors(edge1, edge2);
        double weight = 0.5 / std::tan(angle);
    
        if (boundEMask[i] == 0) {
            // Handle the second face for non-boundary edges
            int idface2 = EF(i,2);
            opposite_vertex = EF(i,3);

            // Calculate vectors for the second angle
            edge1 = V.row(F(idface2,(opposite_vertex + 1) % 3)) - V.row(F(idface2,opposite_vertex));
            edge2 = V.row(F(idface2,(opposite_vertex + 2) % 3)) - V.row(F(idface2,opposite_vertex));

            double angle2 = angle_between_vectors(edge1, edge2);
            double weight2 = 0.5 / std::tan(angle2);
            
            W.coeffRef(i,i) = weight + weight2;
        }
        else {
            W.coeffRef(i,i) = weight;
        }
    }
}

#endif
