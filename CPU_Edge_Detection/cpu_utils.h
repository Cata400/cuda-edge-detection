#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

Mat edge_detect_opencv(Mat image);

void convolve(uchar** input, int height, int width, int kernel_size, float kernel[][3], float** output);

void edge_detect_classic(uchar** input, int height, int width, float** output);