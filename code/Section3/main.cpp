#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/curve_network.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>
#include <set>
#include <array>
#include <queue>
#include "readOFF.h"
#include "create_edge_list.h"
#include "compute_laplacian.h"
#include "tutte_parameterization.h"
#include "lscm_parametrization.h"
#include <chrono>
#include <filesystem>


using namespace Eigen;
using namespace std;

MatrixXi F, E, EF;
VectorXi boundEMask, boundVMask, boundVertices;
MatrixXd V;
SparseMatrix<double> d0, W;
VectorXd vorAreas;
MatrixXd UVBound, UV, UV_LSCM;



int main()
{
    // Define DATA_PATH if not provided by build system
    #ifndef DATA_PATH
    #define DATA_PATH "."
    #endif

    readOFF(DATA_PATH "/param/gargoyle2.off",V, F);
    create_edge_list(F, E, EF, boundEMask, boundVMask, boundVertices, true);
    
    polyscope::init();
    polyscope::SurfaceMesh* psMesh = polyscope::registerSurfaceMesh("Original Mesh", V, F);
    compute_laplacian(V, F, E, EF, boundEMask, d0, W, vorAreas);
    double r = sqrt(vorAreas.sum()/M_PI);
    
    // Compute Tutte Parameterization (with fixed circular boundary)
    auto start = std::chrono::high_resolution_clock::now();
    UVBound = compute_boundary_embedding(V, boundVertices, r);
    UV = compute_tutte_embedding(boundVertices, UVBound, d0, W);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Tutte Parameterization took " << (double)(duration.count())/1000.0 << " seconds to execute." << std::endl;
    
    // Compute LSCM Parameterization (with only two pinned vertices)
    start = std::chrono::high_resolution_clock::now();
    UV_LSCM = lscm_parametrization(V, F);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "LSCM Parameterization took " << (double)(duration.count())/1000.0 << " seconds to execute." << std::endl;
    
    // Setup 3D visualization for Tutte parameterization
    MatrixXd UV3D = MatrixXd::Zero(UV.rows(), 3);
    UV3D.block(0,0,UV.rows(), 1) = UV.col(0);
    UV3D.block(0,2,UV.rows(), 1) = UV.col(1);

    polyscope::SurfaceMesh* psParam = polyscope::registerSurfaceMesh("Tutte Parameterization", UV3D, F)->setEdgeWidth(1.0);
    psMesh->addVertexParameterizationQuantity("Tutte UV Mapping", UV)->setCheckerSize(r/20.0)->setEnabled(true);
    
    // Setup 3D visualization for LSCM parameterization
    MatrixXd UV_LSCM_3D = MatrixXd::Zero(UV_LSCM.rows(), 3);
    UV_LSCM_3D.block(0,0,UV_LSCM.rows(), 1) = UV_LSCM.col(0);
    UV_LSCM_3D.block(0,2,UV_LSCM.rows(), 1) = UV_LSCM.col(1);

    polyscope::SurfaceMesh* psLSCMParam = polyscope::registerSurfaceMesh("LSCM Parameterization", UV_LSCM_3D, F)->setEdgeWidth(1.0);
    psMesh->addVertexParameterizationQuantity("LSCM UV Mapping", UV_LSCM)->setCheckerSize(r/20.0);
    
    vector<RowVector3d> boundNodes;
    vector<RowVector2i> boundEdges;
    
    for (int i=0;i<E.rows();i++){
        if (!boundEMask(i))
            continue;
        
        boundNodes.push_back(V.row(E(i,0)));
        boundNodes.push_back(V.row(E(i,1)));
        RowVector2i boundEdge; boundEdge<<boundNodes.size()-2, boundNodes.size()-1;
        boundEdges.push_back(boundEdge);
    }
    
    polyscope::registerCurveNetwork("Seam", boundNodes, boundEdges)->setRadius(0.0015)->setColor(glm::vec3{0.0,0.0,1.0});
    
    polyscope::show();
    
}

