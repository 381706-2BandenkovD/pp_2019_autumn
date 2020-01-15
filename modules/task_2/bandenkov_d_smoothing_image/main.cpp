// Copyright 2019 Bandenkov Daniil
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include "../../../modules/task_2/bandenkov_d_smoothing_image/smoothing_image.h"

TEST(SMOOTHING_IMAGE, test1_on_wrong_image) {
  EXPECT_ANY_THROW(std::vector <int> matr = getImg(0, -1));
}

TEST(SMOOTHING_IMAGE_MPI, test2_simple_test_par) {
  std::vector <int> matr;
  matr = getImg(5, 7);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    ASSERT_ANY_THROW(ParallelSmoothing(matr, 3, 2));
  }
}

TEST(SMOOTHING_IMAGE_MPI, test2_simple_test_seq) {
  std::vector <int> matr;
  matr = getImg(9, 12);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    ASSERT_ANY_THROW(SequintialSmoothing(matr, 4, 4));
  }
}

TEST(SMOOTHING_IMAGE_MPI, test3_3x3) {
  int row = 3, col = 3;
  std::vector <int> matr(row * col), seq_fin(row * col);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  matr = { 100, 65, 79, 48, 54, 160, 73, 46, 91 };
  seq_fin = { 66, 84, 89, 55, 79, 82, 55, 78, 87 };
  std::vector <int> fin(row * col);
  fin = ParallelSmoothing(matr, row, col);
  if (rank == 0) {
    ASSERT_EQ(seq_fin, fin);
  }
}

TEST(SMOOTHING_IMAGE_MPI, test4_3x4) {
  int row = 3, col = 4;
  std::vector <int> matr(row * col), seq_fin(row * col);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  matr = { 131, 86, 94, 200, 185, 140, 170, 200, 176, 147, 99, 151 };
  seq_fin = { 150, 139, 126, 188, 153, 146, 166, 163, 158, 154, 157, 156 };
  std::vector <int> fin(row * col);
  fin = ParallelSmoothing(matr, row, col);
  if (rank == 0) {
    ASSERT_EQ(seq_fin, fin);
  }
}

TEST(SMOOTHING_IMAGE_MPI, test5_9x15) {
  int row = 9, col = 15;
  std::vector <int> matr(row * col);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  matr = getImg(row, col);
  std::vector <int> fin(row * col);
  fin = ParallelSmoothing(matr, row, col);
  std::vector <int> seq_fin(row * col);
  seq_fin = SequintialSmoothing(matr, row, col);
  if (rank == 0) {
    ASSERT_EQ(seq_fin, fin);
  }
}

TEST(SMOOTHING_IMAGE_MPI, test6_200x200) {
  int row = 200, col = 200;
  std::vector <int> matr(row * col);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  matr = getImg(row, col);
  std::vector <int> fin(row * col);
  fin = ParallelSmoothing(matr, row, col);
  std::vector <int> seq_fin(row * col);
  seq_fin = SequintialSmoothing(matr, row, col);
  if (rank == 0) {
    ASSERT_EQ(seq_fin, fin);
  }
}

// TEST(Smoothing_mpi, test0) {
//  int rank;
//  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//  cv::Mat image;
//  image = imread("C:\\Users\\Archi\\Desktop\\pp.jpg", IMREAD_GRAYSCALE);
//  std::vector <int> a(image.rows * image.cols);
//  if (rank == 0) {
//    cv::namedWindow("My image");
//    imshow("My image", image);
//    for (int i = 0; i < image.rows; i++) {
//      for (int j = 0; j < image.cols; j++) {
//        a[i * image.cols + j] = image.at<uint8_t>(i, j);
//      }
//    }
//  }
//  std::vector <int> ans(image.rows * image.cols), ans_seq(image.rows * image.cols);
//  ans = smoothPar(a, image.rows, image.cols);
//  cv::Mat res = image.clone();
//
//  cv::Mat res_seq = image.clone();
//  if (rank == 0) {
//    ans_seq = smoothSeq(a, image.rows, image.cols, contrast);
//    for (int i = 0; i < image.rows; i++) {
//      for (int j = 0; j < image.cols; j++) {
//        for (int k = 0; k < 3; k++) {
//          res.at<cv::uint8_t>(i, j) = ans[i * image.cols + j];
//          res_seq.at<cv::uint8_t>(i, j) = ans_seq[i * image.cols + j];
//        }
//      }
//    }
//    cv::namedWindow("Sequential Increase Contrast");
//    imshow("Sequential Increase Contrast", res_seq);
//    cv::namedWindow("Parallel Increase Contrast");
//    imshow("Parallel Increase Contrast", res);
//  }
//
//  image.release();
//  res.release();
//  res_seq.release();
//  if (rank == 0) {
//    cv::waitKey(0);
//    ASSERT_EQ(ans_seq, ans);
//  }
//}

TEST(SMOOTHING_IMAGE_MPI, test7_768x768) {
  int row = 768, col = 768;
  std::vector <int> matr(row * col);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  matr = getImg(row, col);
  std::vector <int> fin(row * col);
  fin = ParallelSmoothing(matr, row, col);
  std::vector <int> seq_fin(row * col);
  seq_fin = SequintialSmoothing(matr, row, col);
  if (rank == 0) {
    ASSERT_EQ(seq_fin, fin);
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
