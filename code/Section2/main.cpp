#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>
#include <set>
#include <array>
#include <queue>
#include "readOFF.h"
#include "create_edge_list.h"
#include "compute_angle_defect.h"
#include "compute_laplacian.h"
#include "compute_mean_curvature_normal.h"
#include "compute_areas_normals.h"
#include "mean_curvature_flow.h"
#include <chrono>
#include <filesystem>


using namespace Eigen;
using namespace std;

bool isFlowing = false;
bool isExplicit = true;
double timeStepRatio = 0.5;
double timeStep;

// Flags for measure tracking
bool showGaussianCurvature = false;
bool showMeanCurvature = false;
bool showMeanCurvatureNormal = false;
chrono::steady_clock::time_point lastUpdate = chrono::steady_clock::now();

polyscope::SurfaceMesh* psMesh;

MatrixXi F, E, EF;
VectorXi boundEMask, boundVMask, boundVertices;
MatrixXd origV, currV, Hn, faceNormals;
SparseMatrix<double> d0, W, M, MInv, L;
VectorXd vorAreas, H, faceAreas;

void updateMeasures() {
    if (showGaussianCurvature) {
        VectorXd G = compute_angle_defect(currV, F, boundVMask);
        psMesh->addVertexScalarQuantity("Gaussian Curvature", G.array()/vorAreas.array())->setEnabled(true);
    }
    
    if (showMeanCurvature || showMeanCurvatureNormal) {
        compute_mean_curvature_normal(currV, F, d0.transpose()*W*d0, vorAreas, Hn, H);
        if (showMeanCurvature) {
            psMesh->addVertexScalarQuantity("Mean Curvature", H);
        }
        if (showMeanCurvatureNormal) {
            psMesh->addVertexVectorQuantity("Mean Curvature Normal", Hn);
        }
    }
}

void callback_function() {
    ImGui::PushItemWidth(50);
    
    ImGui::TextUnformatted("Flow Parameters");
    ImGui::Separator();
    bool changed = ImGui::Checkbox("Flow", &isFlowing);
    ImGui::PopItemWidth();
    
    ImGui::Separator();
    ImGui::TextUnformatted("Measures");
    ImGui::Checkbox("Gaussian Curvature", &showGaussianCurvature);
    ImGui::Checkbox("Mean Curvature", &showMeanCurvature);
    ImGui::Checkbox("Mean Curvature Normal", &showMeanCurvatureNormal);
    
    // Update measures every second if any are active
    auto now = chrono::steady_clock::now();
    if (chrono::duration_cast<chrono::seconds>(now - lastUpdate).count() >= 1) {
        if (showGaussianCurvature || showMeanCurvature || showMeanCurvatureNormal) {
            updateMeasures();
        }
        lastUpdate = now;
    }
    
    if (!isFlowing)
        return;
    
    mean_curvature_flow(F, L, timeStep, M, MInv, boundVMask, isExplicit, currV);
    
    psMesh->updateVertexPositions(currV);
    
    // Update measures immediately after flow if they're active
    if (showGaussianCurvature || showMeanCurvature || showMeanCurvatureNormal) {
        updateMeasures();
    }
}

int main()
{
    readOFF(DATA_PATH "/lion-head.off",origV, F);
    currV = origV.rowwise()-origV.colwise().mean();
    create_edge_list(F, E, EF, boundEMask, boundVMask, boundVertices);
    
    polyscope::init();
    psMesh = polyscope::registerSurfaceMesh("Mesh", currV, F);
    compute_areas_normals(origV, F, faceAreas, faceNormals);
    timeStep = timeStepRatio*faceAreas.minCoeff();
    
    compute_laplacian(origV, F, E, EF, boundEMask, d0, W, vorAreas);
    L = d0.transpose()*W*d0;
    vector<Triplet<double>> MTris, MInvTris;
    for (int i=0;i<vorAreas.size();i++){
        MTris.push_back(Triplet<double>(i,i,vorAreas(i)));
        MInvTris.push_back(Triplet<double>(i,i,1.0/vorAreas(i)));
    }
    M.resize(vorAreas.size(), vorAreas.size());
    MInv.resize(vorAreas.size(), vorAreas.size());
    M.setFromTriplets(MTris.begin(), MTris.end());
    MInv.setFromTriplets(MInvTris.begin(), MInvTris.end());
    
    // Initial computation of measures
    VectorXd G = compute_angle_defect(origV, F, boundVMask);
    compute_mean_curvature_normal(origV, F, d0.transpose()*W*d0, vorAreas, Hn, H);
    
    // Register quantities but don't enable them by default
    psMesh->addVertexScalarQuantity("Gaussian Curvature", G.array()/vorAreas.array());
    psMesh->addVertexScalarQuantity("Mean Curvature", H);
    psMesh->addVertexVectorQuantity("Mean Curvature Normal", Hn);
    
    polyscope::state::userCallback = callback_function;
    
    polyscope::show();
}

