#include <iostream>
using namespace std;
typedef struct Matrix {
  int width;
  int height;
  float *elements;
} Mat;

#define BLOCK_SIZE 16

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
void MatMul(const Matrix A, const Matrix B, Matrix C);

int main() {
  Mat h_A;
  h_A.width = 32;
  h_A.height = 32;
  h_A.elements = (float *)malloc(sizeof(float) * h_A.width * h_A.height);
  for (int i = 0; i < h_A.height; ++i) {
    for (int j = 0; j < h_A.width; ++j) {
      h_A.elements[i * h_A.width + j] = 1;
    }
  }
  Mat h_B;
  h_B.width = 32;
  h_B.height = 32;
  h_B.elements = (float *)malloc(sizeof(float) * h_B.width * h_B.height);

  for (int i = 0; i < h_B.height; ++i) {
    for (int j = 0; j < h_B.width; ++j) {
      h_B.elements[i * h_B.width + j] = 2;
    }
  }

  Mat h_C;
  h_C.width = 32;
  h_C.height = 32;
  h_C.elements = (float *)malloc(sizeof(float) * h_C.width * h_C.height);

  for (int i = 0; i < h_C.height; ++i) {
    for (int j = 0; j < h_C.width; ++j) {
      h_C.elements[i * h_C.width + j] = 0;
    }
  }
  MatMul(h_A, h_B, h_C);
  float sum_error = 0.0;

  for (int i = 0; i < h_C.height; ++i) {
    for (int j = 0; j < h_C.width; ++j) {
      sum_error += fabs(64 - h_C.elements[i * h_C.width + j]);
    }
  }
  cout << "sum error : " << sum_error << endl;
  free(h_A.elements);
  free(h_B.elements);
  free(h_C.elements);
  return 0;
}
void MatMul(const Matrix A, const Matrix B, Matrix C) {
  Mat d_A;
  d_A.width = A.width;
  d_A.height = A.height;
  size_t size = A.width * A.height * sizeof(float);
  cudaMalloc((void **)&d_A.elements, size);
  cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

  Mat d_B;
  d_B.width = B.width;
  d_B.height = B.height;
  size = B.width * B.height * sizeof(float);
  cudaMalloc((void **)&d_B.elements, size);
  cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

  Mat d_C;
  d_C.width = C.width;
  d_C.height = C.height;
  size = C.width * C.height * sizeof(float);
  cudaMalloc((void **)&d_C.elements, size);

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
  MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

  cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
  cudaFree(d_A.elements);
  cudaFree(d_B.elements);
  cudaFree(d_C.elements);
}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
  float CValue = 0;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  for (int e = 0; e < A.width; ++e) {
    CValue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
  }
  C.elements[row * C.width + col] = CValue;
}
