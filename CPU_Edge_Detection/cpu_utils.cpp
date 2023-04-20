#include "cpu_utils.h"

Mat edge_detect_opencv(Mat input) {
	Mat grad_x, grad_y;
	Sobel(input, grad_x, CV_32F, 1, 0, 3, 1, 0, BORDER_CONSTANT);
	Sobel(input, grad_y, CV_32F, 0, 1, 3, 1, 0, BORDER_CONSTANT);

	Mat grad;
	pow(grad_x, 2, grad_x);
	pow(grad_y, 2, grad_y);
	sqrt(grad_x + grad_y, grad);

	return grad;
}

cuda::GpuMat edge_detect_opencv_gpu(cuda::GpuMat input) {
	cuda::GpuMat grad_x, grad_y;

	Ptr<cuda::Filter> sobel_x = cuda::createSobelFilter(input.type(), CV_32F, 1, 0, 3, 1, BORDER_CONSTANT, BORDER_CONSTANT);
	Ptr<cuda::Filter> sobel_y = cuda::createSobelFilter(input.type(), CV_32F, 0, 1, 3, 1, BORDER_CONSTANT, BORDER_CONSTANT);

	sobel_x->apply(input, grad_x);
	sobel_y->apply(input, grad_y);

	cuda::GpuMat grad;
	cuda::pow(grad_x, 2, grad_x);
	cuda::pow(grad_y, 2, grad_y);
	cuda::add(grad_x, grad_y, grad);
	cuda::sqrt(grad, grad);

	return grad;
}


void convolve(uchar** input, int height, int width, int kernel_size, float kernel[][3], float** output) {
	int half_kernel_size = kernel_size / 2;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float sum = 0;
			for (int k = 0; k < kernel_size; k++) {
				for (int l = 0; l < kernel_size; l++) {
					int x = i + k - half_kernel_size;
					int y = j + l - half_kernel_size;
					if (x < 0 || x >= height || y < 0 || y >= width) {
						continue;
					}
					sum += input[x][y] * kernel[k][l];
				}
			}
			output[i][j] = sum;
		}
	}
}

void edge_detect_classic(uchar** input, int height, int width, float** output) {
	const int kernel_size = 3;
	float gx[kernel_size][kernel_size] = { { 1, 0, -1 }, { 2, 0, -2 }, { 1, 0, -1 } };
	float gy[kernel_size][kernel_size] = { { 1, 2, 1 },  { 0, 0, 0 }, { -1, -2, -1 }};

	float** grad_x = new float*[height];
	float** grad_y = new float*[height];
	for (int i = 0; i < height; i++) {
		grad_x[i] = new float[width];
		grad_y[i] = new float[width];
	}

	convolve(input, height, width, kernel_size, gx, grad_x);
	convolve(input, height, width, kernel_size, gy, grad_y);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			output[i][j] = sqrt(grad_x[i][j] * grad_x[i][j] + grad_y[i][j] * grad_y[i][j]);
		}
	}

	for (int i = 0; i < height; i++) {
		delete[] grad_x[i];
		delete[] grad_y[i];
	}

	delete[] grad_x;
	delete[] grad_y;
}


