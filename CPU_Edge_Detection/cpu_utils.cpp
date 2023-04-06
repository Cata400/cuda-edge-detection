#include "cpu_utils.h"

Mat edge_detect_opencv(Mat image) {
	Mat grad_x, grad_y;
	Sobel(image, grad_x, CV_32F, 1, 0, 3, 1, 0, BORDER_CONSTANT);
	Sobel(image, grad_y, CV_32F, 0, 1, 3, 1, 0, BORDER_CONSTANT);

	Mat grad;
	pow(grad_x, 2, grad_x);
	pow(grad_y, 2, grad_y);
	sqrt(grad_x + grad_y, grad);

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
	float gx_kernel[kernel_size][kernel_size] = { { 1, 0, -1 }, { 2, 0, -2 }, { 1, 0, -1 } };
	float gy_kernel[kernel_size][kernel_size] = { { 1, 2, 1 },  { 0, 0, 0 }, { -1, -2, -1 }};

	float** gx = new float*[height];
	float** gy = new float*[height];
	for (int i = 0; i < height; i++) {
		gx[i] = new float[width];
		gy[i] = new float[width];
	}

	convolve(input, height, width, kernel_size, gx_kernel, gx);
	convolve(input, height, width, kernel_size, gy_kernel, gy);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			output[i][j] = sqrt(gx[i][j] * gx[i][j] + gy[i][j] * gy[i][j]);
		}
	}

	for (int i = 0; i < height; i++) {
		delete[] gx[i];
		delete[] gy[i];
	}

	delete[] gx;
	delete[] gy;
}

