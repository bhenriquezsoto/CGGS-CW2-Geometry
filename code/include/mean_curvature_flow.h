#ifndef COMPUTE_MEAN_CURVATURE_FLOW_HEADER_FILE
#define COMPUTE_MEAN_CURVATURE_FLOW_HEADER_FILE

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "compute_areas_normals.h"

// Helper function to compute total mesh area from face areas
double compute_total_area(const Eigen::MatrixXi& F, const Eigen::MatrixXd& V) {
    Eigen::VectorXd faceAreas;
    Eigen::MatrixXd faceNormals;
    compute_areas_normals(V, F, faceAreas, faceNormals);
    return faceAreas.sum();
}

void mean_curvature_flow(const Eigen::MatrixXi& F,
                         const Eigen::SparseMatrix<double>& L,
                         const double timeStep,
                         const Eigen::SparseMatrix<double>& M,
                         const Eigen::SparseMatrix<double>& MInv,
                         const Eigen::VectorXi& boundVMask,
                         const bool isExplicit,
                         Eigen::MatrixXd& currV){
    
    using namespace Eigen;
    using namespace std;

    // Store previous positions for boundary vertex handling
    MatrixXd previousPositions(currV);
    
    // Calculate initial mesh area (before flow) to rescale the mesh
    double previous_area = compute_total_area(F, currV);
    
    // Check if mesh has boundaries
    bool hasBoundary = (boundVMask.sum() > 0);
    
    if (isExplicit){
        currV = currV - timeStep * MInv * L * currV;
    }
    else {
        SparseMatrix<double> left_side = M + timeStep * L;
        SimplicialLDLT<SparseMatrix<double>> solver;
        solver.compute(left_side);
        MatrixXd right_side = M * currV;
        currV = solver.solve(right_side);
    }

    // Fix boundary vertices by snapping them back to their previous positions
    for (int i = 0; i < currV.rows(); i++){
        if (boundVMask(i) == 1){
            currV.row(i) = previousPositions.row(i);
        }
    }
    
    if (!hasBoundary) {
        // Scale mesh to preserve area
        double current_area = compute_total_area(F, currV);
        double scale_factor = sqrt(previous_area / current_area);
        currV *= scale_factor;
        
        // Center mesh to have zero average
        Vector3d center = currV.colwise().sum() / currV.rows();
        currV = currV.rowwise() - center.transpose();
    }
}

#endif
