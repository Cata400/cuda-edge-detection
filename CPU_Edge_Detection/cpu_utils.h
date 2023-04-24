#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <cmath>
#include <iostream>

using namespace cv;

Mat sobel_opencv(Mat input);

Mat canny_opencv(Mat input, int low_tr, int high_tr, int aperture);

cuda::GpuMat sobel_opencv_gpu(cuda::GpuMat input);

cuda::GpuMat canny_opencv_gpu(cuda::GpuMat input, int low_tr, int high_tr, int aperture);

template <typename T>
void convolve(uchar** input, int height, int width, int kernel_size, float** kernel, T** output);

void sobel_classic(uchar** input, int height, int width, float** output, float** orientations=NULL);

template <typename T>
void fix_borders(T** input, int height, int width, int kernel_size);

void normalize_image(float** input, int height, int width, uchar** output);

void non_max_suppression(uchar** grad_norm, int height, int width, float** orientations, uchar** grad_suppressed);

void hysterezis(uchar** input, int height, int width);

void canny_classic(uchar** input, int height, int width, int low_tr, int high_tr, uchar** output);