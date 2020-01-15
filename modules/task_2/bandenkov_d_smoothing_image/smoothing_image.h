// Copyright 2019 Bandenkov Daniil
#ifndef MODULES_TASK_2_BANDENKOV_D_SMOOTHING_IMAGE_SMOOTHING_IMAGE_H_
#define MODULES_TASK_2_BANDENKOV_D_SMOOTHING_IMAGE_SMOOTHING_IMAGE_H_

#include <vector>

std::vector<int> getImg(const int  rows, const int cols);
std::vector<int> SequintialSmoothing(const std::vector<int>& img, int rows, int cols);
std::vector<int> ParallelSmoothing(const std::vector<int>& img, int rows, int cols);
#endif  // MODULES_TASK_2_BANDENKOV_D_SMOOTHING_IMAGE_SMOOTHING_IMAGE_H_
