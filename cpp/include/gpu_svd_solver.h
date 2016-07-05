// The MIT License (MIT)
//
// Copyright (c) 2016 Northeastern University
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef CPP_INCLUDE_GPU_SVD_SOLVER_H_
#define CPP_INCLUDE_GPU_SVD_SOLVER_H_

#ifdef NEED_CUDA
#include<cuda_runtime_api.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<cusolverDn.h>

#include<iostream>

#include "Eigen/Dense"
#include "include/matrix.h"
#include "include/vector.h"
#include "include/gpu_util.h"

namespace Nice {

cusolverStatus_t doSvd(cusolverDnHandle_t solver_handle,
           int M,
           int N,
           float * d_A,
           float * d_S,
           float * d_U,
           float * d_V,
           float * work,
           int work_size,
           int * devInfo) {
  return cusolverDnSgesvd(solver_handle,
                          'A', 'A', M, N, d_A, M, d_S,
                          d_U, M, d_V, N, work, work_size,
                          NULL, devInfo);
}

cusolverStatus_t doSvd(cusolverDnHandle_t solver_handle,
           int M,
           int N,
           double * d_A,
           double * d_S,
           double * d_U,
           double * d_V,
           double * work,
           int work_size,
           int * devInfo) {
  return cusolverDnDgesvd(solver_handle,
                         'A', 'A', M, N, d_A, M, d_S,
                          d_U, M, d_V, N, work, work_size,
                          NULL, devInfo);
}

template<typename T>
class GpuSvdSolver {
 private:
  Matrix<T> u_;
  Matrix<T> v_;
  Vector<T> s_;

 public:
  GpuSvdSolver() {}

  void Compute(const Matrix<T> &A) {
    int M = A.rows();
    int N = A.cols();
    const T *h_A = &A(0);
    // --- Setting the device matrix and moving the host matrix to the device
    T *d_A;   gpuErrchk(cudaMalloc(&d_A,      M * N * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_A, h_A, M * N * sizeof(T), cudaMemcpyHostToDevice));

    //--- host side SVD results space
    s_.resize(M, 1);
    u_.resize(M, M);
    v_.resize(N, N);

    // --- device side SVD workspace and matrices
    int work_size = 0;
    int *devInfo;   gpuErrchk(cudaMalloc(&devInfo,          sizeof(int)));
    T *d_U;         gpuErrchk(cudaMalloc(&d_U,      M * M * sizeof(T)));
    T *d_V;         gpuErrchk(cudaMalloc(&d_V,      N * N * sizeof(T)));
    T *d_S;         gpuErrchk(cudaMalloc(&d_S,      N *     sizeof(T)));
    cusolverStatus_t stat;

    // --- CUDA solver initialization
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);
    stat = cusolverDnSgesvd_bufferSize(solver_handle, M, N, &work_size);
    if (stat != CUSOLVER_STATUS_SUCCESS)
      std::cout << "Initialization of cuSolver failed. \n";
    T *work;    gpuErrchk(cudaMalloc(&work, work_size * sizeof(T)));

    // --- CUDA SVD execution
    stat = doSvd(solver_handle, M, N,
                 d_A, d_S, d_U, d_V,
                 work, work_size, devInfo);
    cudaDeviceSynchronize();

    int devInfo_h = 0;
    gpuErrchk(cudaMemcpy(&devInfo_h, devInfo,
              sizeof(int), cudaMemcpyDeviceToHost));

    // --- Moving the results from device to host
    gpuErrchk(cudaMemcpy(&s_(0, 0), d_S, N*sizeof(T),
              cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&u_(0, 0), d_U, M*M*sizeof(T),
              cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&v_(0, 0), d_V, N*N*sizeof(T),
              cudaMemcpyDeviceToHost));

    cudaFree(d_S); cudaFree(d_U); cudaFree(d_V);
    cusolverDnDestroy(solver_handle);
  }

  Matrix<T> MatrixU() const              { return u_; }

  Matrix<T> MatrixV() const              { return v_; }

  Vector<T> SingularValues() const       { return s_; }
};
}  // namespace Nice

#endif  // NEED_CUDA

#endif  // CPP_INCLUDE_GPU_SVD_SOLVER_H_

