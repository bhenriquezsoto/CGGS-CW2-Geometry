#ifndef COMPUTE_TUTTE_PARAMETERIZATION_HEADER_FILE
#define COMPUTE_TUTTE_PARAMETERIZATION_HEADER_FILE

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "slice_columns_sparse.h"
#include "set_diff.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Eigen::MatrixXd compute_boundary_embedding(const Eigen::MatrixXd& V,
                                           const Eigen::VectorXi& boundVertices,
                                           const double r){
    
    double sumRelativeLengths = 0;
    for(int i = 0; i < boundVertices.size() - 1; i++){
        sumRelativeLengths += (V.row(boundVertices(i+1)) - V.row(boundVertices(i))).norm();
    }
    sumRelativeLengths += (V.row(boundVertices(0)) - V.row(boundVertices(boundVertices.size()-1))).norm();

    Eigen::VectorXd sectorAngles(boundVertices.size());
    for(int i = 0; i < boundVertices.size() - 1; i++){
        sectorAngles(i) = 2 * M_PI * (V.row(boundVertices(i+1)) - V.row(boundVertices(i))).norm() / sumRelativeLengths;
    }
    sectorAngles(boundVertices.size()-1) = 2 * M_PI * (V.row(boundVertices(0)) - V.row(boundVertices(boundVertices.size()-1))).norm() / sumRelativeLengths;

    Eigen::MatrixXd embedding(boundVertices.size(),2);
    embedding(0,0) = r;
    embedding(0,1) = 0;
    for(int i = 1; i < boundVertices.size(); i++){
        embedding(i,0) = r * cos(sectorAngles.head(i).sum());
        embedding(i,1) = r * sin(sectorAngles.head(i).sum());
    }
    return embedding;
    
}

Eigen::MatrixXd compute_tutte_embedding(const Eigen::VectorXi& boundVertices,
                                        const Eigen::MatrixXd& UVBound,
                                        const Eigen::SparseMatrix<double>& d0,
                                        const Eigen::SparseMatrix<double>& W){
    
    // Create vector of all vertex indices
    Eigen::VectorXi allVertices(d0.cols());
    for (int i = 0; i < d0.cols(); i++) {
        allVertices(i) = i;
    }
    
    // Find interior vertices and slice d0 matrix
    Eigen::VectorXi interiorVertices = set_diff(allVertices, boundVertices);
    Eigen::SparseMatrix<double> d0_I, d0_B;
    d0_I = slice_columns_sparse(d0, interiorVertices);
    d0_B = slice_columns_sparse(d0, boundVertices);
    Eigen::SparseMatrix<double> A = d0_I.transpose() * W * d0_I;
    
    // Initialize result matrix with correct size (total number of vertices)
    Eigen::MatrixXd UV = Eigen::MatrixXd::Zero(d0.cols(), 2);
    for (int i = 0; i < boundVertices.size(); i++) {
        UV.row(boundVertices(i)) = UVBound.row(i);
    }
    
    // Solve for each coordinate (x and y) separately
    for (int coord = 0; coord < 2; coord++) {
        // Extract coordinate column from UVBound
        Eigen::VectorXd UVBound_coord = UVBound.col(coord);
        
        // Compute right-hand side vector b
        Eigen::VectorXd b = -d0_I.transpose() * W * d0_B * UVBound_coord;
        
        // Solve system
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(A);
        Eigen::VectorXd x = solver.solve(b);
        
        // Properly place solution in UV
        for (int i = 0; i < interiorVertices.size(); i++) {
            UV(interiorVertices(i), coord) = x(i);
        }
    }
    
    return UV;
}

#endif
