/* This code was modified using the base code from:
https://github.com/math-castro/lscm/blob/master/src/parametrization/parametrization.cpp
*/

#ifndef COMPUTE_LSCM_PARAMETERIZATION_HEADER_FILE
#define COMPUTE_LSCM_PARAMETERIZATION_HEADER_FILE

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include "slice_columns_sparse.h"
#include "set_diff.h"
#include <vector>
#include <cmath>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace Eigen;
using namespace std;

typedef Triplet<double> Td;

void emplaceAB(vector<Td> &ta, vector<Td> &tb, int i, int j, int n, int np, double wr, double wi, pair<int,int> &diam) {
  if (j != diam.first and j != diam.second) {
    if(j > diam.second) j--;
    if(j > diam.first) j--;
    ta.emplace_back(i, j, wr);
    ta.emplace_back(i + np, j + n - 2, wr);
    ta.emplace_back(i, j + n - 2, -wi);
    ta.emplace_back(i + np, j, wi);
  } else {
    j = (j==diam.second);
    tb.emplace_back(i, j, wr);
    tb.emplace_back(i + np, j + 2, wr);
    tb.emplace_back(i, j + 2, -wi);
    tb.emplace_back(i + np, j, wi);
  }
}

double angleBetweenSides(const RowVectorXd &a, const RowVectorXd &b) {
  return acos(a.dot(b) / a.norm() / b.norm());
}

// The fixed vertex are the two vertices with the largest distance to each other
// This in order to set the correct frame of reference for the parametrization
// But there are other choices, for example the two vertices with the smallest angle between them
pair<int,int> approximateDiameter(const MatrixXd &V) {
  const int n = V.rows();
  double m = 0;
  pair<int,int> best{0,1};
  for(int i = 0; i < n; i++) {
    for(int j = i+1; j < n; j++) {
      double d = (V.row(i)-V.row(j)).norm();
      if(d>m) {
        m = d;
        best = pair<int,int>{i,j};
      }
    }
  }
  return best;
}

MatrixXd lscm_parametrization(const MatrixXd &V, const MatrixXi &F) {
  const int np = F.rows(), n = V.rows();

  pair<int,int> diam = approximateDiameter(V);

  // List of triplets for sparse matrix creation
  vector<Td> ta, tb, tu;
  ta.reserve(np * 12);

  // For each triangle
  for (int i = 0; i < np; i++) {
    // Calculate x,y
    // First vertex as (0,0), second as (s,0), third as t(cos(a), sin(a))
    RowVectorXd p[3];
    double x[3], y[3];

    p[0] = V.row(F(i, 0));
    p[1] = V.row(F(i, 1));
    p[2] = V.row(F(i, 2));

    double a = angleBetweenSides(p[1] - p[0], p[2] - p[0]);
    double s = (p[1] - p[0]).norm();
    double t = (p[2] - p[0]).norm();

    x[0] = y[0] = 0;
    x[1] = s, y[1] = 0;
    x[2] = t * cos(a), y[2] = t * sin(a);

    double d = sqrt((x[0] * y[1] - y[0] * x[1]) + (x[1] * y[2] - y[1] * x[2]) + (x[2] * y[0] - y[2] * x[0]));
//    d=1;

    // Calculate W
    double Wr[3], Wi[3];
    Wr[0] = (x[2] - x[1]) / d, Wi[0] = (y[2] - y[1]) / d;
    Wr[1] = (x[0] - x[2]) / d, Wi[1] = (y[0] - y[2]) / d;
    Wr[2] = (x[1] - x[0]) / d, Wi[2] = (y[1] - y[0]) / d;


    // Push values to A and B
    emplaceAB(ta, tb, i, F(i,0), n, np, Wr[0], Wi[0], diam);
    emplaceAB(ta, tb, i, F(i,1), n, np, Wr[1], Wi[1], diam);
    emplaceAB(ta, tb, i, F(i,2), n, np, Wr[2], Wi[2], diam);
  }

  // Build A and B
  SparseMatrix<double> A(2 * np, 2 * (n - 2));
  SparseMatrix<double> B(2 * np, 4);
  A.setFromTriplets(ta.begin(), ta.end());
  ta.clear();
  B.setFromTriplets(tb.begin(), tb.end());
  tb.clear();

  // Build up 
  VectorXd up(4);
  up(0) = 0;
  up(1) = 1;
  up(2) = 0;
  up(3) = 1;

  VectorXd b = - B * up;

  // Solve least squares using QR decomposition
  // A.makeCompressed();
  // SparseQR<SparseMatrix<double>, COLAMDOrdering<int> > solver(A);
  // VectorXd uf = solver.solve(b);

  // Solve least squares using iterative CG
  A.makeCompressed();
  LeastSquaresConjugateGradient<SparseMatrix<double>> solver(A);
  VectorXd uf = solver.solve(b);


  // Join uf and up
  MatrixXd U(n, 2);
  for(int i = 0; i < n; i++) {
    if(i != diam.first and i != diam.second) {
      int ii = i;
      if(i > diam.second) ii--;
      if(i > diam.first) ii--;
      U(i,0) = uf(ii);
      U(i,1) = uf(ii + n - 2);
    }
    else
      U(i,0) = U(i,1) = (i==diam.second);
  }

  return U;
}

#endif
