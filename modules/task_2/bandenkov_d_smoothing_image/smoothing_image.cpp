// Copyright 2019 Bandenkov Daniil
#include <mpi.h>
#include <vector>
#include <random>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include "../../../modules/task_2/bandenkov_d_smoothing_image/smoothing_image.h"

std::vector<int> getImg(const int rows, const int cols) {
  if (rows < 0 || cols < 0)
    throw - 1;
  std::mt19937 gen;
  gen.seed(static_cast<unsigned int>(time(0)));
  std::vector<int> Matrix(rows * cols);
  for (int i = 0; i < rows * cols; i++)
    Matrix[i] = gen() % 256;
  return Matrix;
}

std::vector<int> SequintialSmoothing(const std::vector<int>& img, int rows, int cols) {
  if (static_cast<int>(img.size()) != rows * cols) {
    throw - 1;
  }
  if (rows < 0 || cols < 0)
    throw - 1;

  std::vector<int> smothImg(rows * cols);
  for (int i = 0; i < rows * cols; i++) {
    int sum = img[i];
    int n = 1;
    int coeff = 0;
    if (i % rows) {
      sum += img[i - 1];
      n++;
      coeff++;
    }
    if ((i + 1) % rows) {
      sum += img[i + 1];
      n++;
      coeff++;
    }
    if (i > rows) {
      sum += img[i - rows];
      n++;
      coeff++;
      if (i % rows) {
        sum += img[i - rows - 1];
        n++;
        coeff++;
      }
      if ((i + 1) % rows) {
        sum += img[i - rows + 1];
        n++;
        coeff++;
      }
    }
    if (i / rows + 1 < cols) {
      sum += img[i + rows];
      n++;
      coeff++;
      if (i % rows) {
        sum += img[i + rows - 1];
        n++;
        coeff++;
      }
      if ((i + 1) % rows) {
        sum += img[i + rows + 1];
        n++;
        coeff++;
      }
    }
    smothImg[i] = sum / n;
  }
  return smothImg;
}

std::vector<int> ParallelSmoothing(const std::vector<int>& img, int rows, int cols) {
  if (static_cast<int>(img.size()) != rows * cols) {
    throw - 1;
  }
  if (rows < 0 || cols < 0)
    throw - 1;

  int size, rank;
  std::vector<int> smothImg(rows * cols);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int det = rows * cols / size;
  int rem = rows * cols % size;

  std::vector<int> startPos(size);
  for (int i = 1; i < size; i++)
    startPos[i] = i * det + rem;
  int cf = 0;
  if (rank == 0) {
    if (det > 0) {
      for (int proc = 1; proc < size; proc++) {
        MPI_Send(&startPos[proc], 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
        cf++;
      }
    }
  }
  else {
    if (det > 0) {
      int startPos = -1;
      MPI_Status status;
      MPI_Recv(&startPos, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
      for (int i = startPos; i < startPos + det; i++) {
        int sum = img[i];
        int n = 1;
        int coeff = 0;
        if (i % rows) {
          sum += img[i - 1];
          n++;
          coeff++;
        }
        if ((i + 1) % rows) {
          sum += img[i + 1];
          n++;
          coeff++;
        }
        if (i > rows) {
          sum += img[i - rows];
          n++;
          coeff++;
          if (i % rows) {
            sum += img[i - rows - 1];
            n++;
            coeff++;
          }
          if ((i + 1) % rows) {
            sum += img[i - rows + 1];
            n++;
            coeff++;
          }
        }
        if (i / rows + 1 < cols) {
          sum += img[i + rows];
          n++;
          coeff++;
          if (i % rows) {
            sum += img[i + rows - 1];
            n++;
            coeff++;
          }
          if ((i + 1) % rows) {
            sum += img[i + rows + 1];
            n++;
            coeff++;
          }
        }
        cf += coeff;
        smothImg[i] = sum / n;
      }
      MPI_Send(&smothImg[startPos], det, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
  }
  if (rank == 0) {

    for (int i = 0; i < det + rem; i++) {
      int sum = img[i];
      int n = 1;
      int coeff = 0;
      if (i % rows) {
        sum += img[i - 1];
        n++;
        coeff++;
      }
      if ((i + 1) % rows) {
        sum += img[i + 1];
        n++;
        coeff++;
      }
      if (i > rows) {
        sum += img[i - rows];
        n++;
        coeff++;
        if (i % rows) {
          sum += img[i - rows - 1];
          n++;
          coeff++;
        }
        if ((i + 1) % rows) {
          sum += img[i - rows + 1];
          n++;
          coeff++;
        }
      }
      if (i / rows + 1 < cols) {
        sum += img[i + rows];
        n++;
        coeff++;
        if (i % rows) {
          sum += img[i + rows - 1];
          n++;
          coeff++;
        }
        if ((i + 1) % rows) {
          sum += img[i + rows + 1];
          n++;
          coeff++;
        }
      }
      cf += coeff;
      smothImg[i] = sum / n;
    }
    MPI_Status status;
    for (int j = 1; j < size; j++) {
      MPI_Recv(&smothImg[startPos[j]], det, MPI_INT, j, 0, MPI_COMM_WORLD, &status);
    }
  }
  return smothImg;
}
