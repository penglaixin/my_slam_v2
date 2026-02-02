#pragma once

// C++ 标准库
#include <iostream>
#include <vector>
#include <string>
#include <deque>
#include <memory> // 智能指针

// OpenCV
#include <opencv2/opencv.hpp>

// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>

// G2O
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sba/types_sba.h>
#include <g2o/core/robust_kernel_impl.h>

using namespace std;