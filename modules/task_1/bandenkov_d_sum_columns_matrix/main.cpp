// Copyright 2019 Bandenkov Daniil
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include "../../../modules/task_1/bandenkov_d_sum_columns_matrix/sum_columns.h"

TEST(SUM_COLUMNS_MPI, test1_on_wrong_matrix) {
  EXPECT_ANY_THROW(std::vector <int> a = getRandomMatrix(0, 2));
}

TEST(SUM_COLUMNS_MPI, test2_getSequintialSum_matrix2x2) {
  std::vector<int> global_mat(4);
  global_mat = { 1, 2, 3, 4 };
  std::vector<int> sequintial_sum = getSequintialSum(global_mat, 2, 2);
  std::vector<int> res(2);
  res = { 3, 7 };
  ASSERT_EQ(res, sequintial_sum);
}

TEST(SUM_COLUMNS_MPI, test3_square_matrix2x2) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<int> global_mat(4);
  global_mat = { 1, 2, 3, 4 };
  std::vector<int> parallel_sum = getParallelSum(global_mat, 2, 2);
  std::vector<int> sequintial_sum = getSequintialSum(global_mat, 2, 2);
  if (rank == 0) {
    ASSERT_EQ(sequintial_sum, parallel_sum);
  }
}

TEST(SUM_COLUMNS_MPI, test4_square_matrix_100x100) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<int> rand_matrix;
  const int rows = 100;
  const int cols = rows;
  rand_matrix = getRandomMatrix(rows, cols);
  std::vector<int> trans_matrix(rows * cols);
  trans_matrix = GetTransposeMatrix(rand_matrix, rows, cols);
  std::vector<int> parallel_sum = getParallelSum(trans_matrix, rows, cols);
  std::vector<int> sequintial_sum = getSequintialSum(trans_matrix, rows, cols);
  if (rank == 0) {
    ASSERT_EQ(sequintial_sum, parallel_sum);
  }
}

TEST(SUM_COLUMNS_MPI, test5_rectangle_matrix_80x60) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<int> rand_matrix;
  const int rows = 80;
  const int cols = 60;
  rand_matrix = getRandomMatrix(rows, cols);
  std::vector<int> trans_matrix(rows * cols);
  trans_matrix = GetTransposeMatrix(rand_matrix, rows, cols);
  std::vector<int> parallel_sum = getParallelSum(trans_matrix, rows, cols);
  std::vector<int> sequintial_sum = getSequintialSum(trans_matrix, rows, cols);
  if (rank == 0) {
    ASSERT_EQ(sequintial_sum, parallel_sum);
  }
}

TEST(SUM_COLUMNS_MPI, test5_1_rectangle_const_matrix4x2) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<int> matrix(8);
  const int rows = 4;
  const int cols = 2;
  matrix = { 1, 2, 3, 4, 5, 6, 7, 8 };
  std::vector<int> parallel_sum = getParallelSum(matrix, rows, cols);
    std::vector<int> sequintial_sum = getSequintialSum(matrix, rows, cols);
  if (rank == 0) {
    ASSERT_EQ(sequintial_sum, parallel_sum);
  }
}
TEST(SUM_COLUMNS_MPI, test5_2_rectangle_const_matrix4x3) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<int> matrix(12);
  const int rows = 4;
  const int cols = 3;
  matrix = { 1, 2, 3, 4, 0, 0, 0, 0, 1, 1, 1, 1 };
  std::vector<int> parallel_sum = getParallelSum(matrix, rows, cols);
  std::vector<int> sequintial_sum = getSequintialSum(matrix, rows, cols);
  if (rank == 0) {
    ASSERT_EQ(sequintial_sum, parallel_sum);
  }
}
TEST(SUM_COLUMNS_MPI, test5_3_rectangle_const_matrix4x3) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<int> matrix(12);
  const int rows = 4;
  const int cols = 3;
  matrix = { 1, 0, 0, 5, 0, 0, 0, 1, 1, 1, 1, 1 };
  std::vector<int> parallel_sum = getParallelSum(matrix, rows, cols);
  std::vector<int> sequintial_sum = getSequintialSum(matrix, rows, cols);
  if (rank == 0) {
    ASSERT_EQ(sequintial_sum, parallel_sum);
  }
}


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
  ::testing::TestEventListeners& listeners =
    ::testing::UnitTest::GetInstance()->listeners();

  listeners.Release(listeners.default_result_printer());
  listeners.Release(listeners.default_xml_generator());

  listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);

  return RUN_ALL_TESTS();
}
