#include "cpu_utils.h"

Mat sobel_opencv(Mat input) {
	Mat grad_x, grad_y;
	Sobel(input, grad_x, CV_32F, 1, 0, 3, 1, 0, BORDER_CONSTANT);
	Sobel(input, grad_y, CV_32F, 0, 1, 3, 1, 0, BORDER_CONSTANT);

	Mat grad;
	pow(grad_x, 2, grad_x);
	pow(grad_y, 2, grad_y);
	sqrt(grad_x + grad_y, grad);

	return grad;
}

Mat canny_opencv(Mat input, int low_tr, int high_tr, int aperture) {
	Mat edges;
	Canny(input, edges, low_tr, high_tr, aperture, true);
	return edges;
}

cuda::GpuMat sobel_opencv_gpu(cuda::GpuMat input) {
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

cuda::GpuMat canny_opencv_gpu(cuda::GpuMat input, int low_tr, int high_tr, int aperture) {
	cuda::GpuMat edges;

	Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector(low_tr, high_tr, aperture, true);
	canny->detect(input, edges);

	return edges;
}

template <typename T>
void convolve(uchar** input, int height, int width, int kernel_size, float** kernel, T** output) {
	int half_kernel_size = kernel_size / 2;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			T sum = 0;
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

void sobel_classic(uchar** input, int height, int width, float** output, float** orientations) {
	const int kernel_size = 3;

	float gx_elements[kernel_size * kernel_size] = { 1, 0, -1, 2, 0, -2, 1, 0, -1 };
	float gy_elements[kernel_size * kernel_size] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

	float** gx = new float*[kernel_size];
	float** gy = new float*[kernel_size];

	for (int i = 0; i < kernel_size; i++) {
		gx[i] = new float[kernel_size];
		gy[i] = new float[kernel_size];
	}

	for (int i = 0; i < kernel_size; i++) {
		for (int j = 0; j < kernel_size; j++) {
			gx[i][j] = gx_elements[i * kernel_size + j];
			gy[i][j] = gy_elements[i * kernel_size + j];
		}
	}

	float** grad_x = new float*[height];
	float** grad_y = new float*[height];
	for (int i = 0; i < height; i++) {
		grad_x[i] = new float[width];
		grad_y[i] = new float[width];
	}

	convolve<float>(input, height, width, kernel_size, gx, grad_x);
	convolve<float>(input, height, width, kernel_size, gy, grad_y);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			output[i][j] = sqrt(grad_x[i][j] * grad_x[i][j] + grad_y[i][j] * grad_y[i][j]);
			if (orientations != NULL) {
				orientations[i][j] = atan2(grad_y[i][j], grad_x[i][j]);
			}
		}
	}

	for (int i = 0; i < height; i++) {
		delete[] grad_x[i];
		delete[] grad_y[i];
	}

	delete[] grad_x;
	delete[] grad_y;

	for (int i = 0; i < kernel_size; i++) {
		delete[] gx[i];
		delete[] gy[i];
	}

	delete[] gx;
	delete[] gy;
}

template <typename T>
void fix_borders(T** input, int height, int width, int kernel_size) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < kernel_size / 2; j++) {
			input[i][j] = input[i][kernel_size / 2];
			input[i][width - 1 - j] = input[i][width - 1 - kernel_size / 2];
		}

	}
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < kernel_size / 2; j++) {
			input[j][i] = input[kernel_size / 2][i];
			input[height - 1 - j][i] = input[height - 1 - kernel_size / 2][i];
		}
	}
}


void normalize_image(float** input, int height, int width, uchar** output) {
	float max = 0, min = 255;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			max = max < input[i][j] ? input[i][j] : max;
			min = min > input[i][j] ? input[i][j] : min;
		}
	}

	float range = max - min;
	float scale = 255 / range;

	// Normalize
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			output[i][j] = (uchar)((input[i][j] - min) * scale);
		}
	}
}

void non_max_suppression(uchar** grad_norm, int height, int width, float** orientations, uchar** grad_suppressed) {
	float angle;

	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			int q = 255, r = 255;

			angle = orientations[i][j] * 180 / M_PI;
			angle = angle < 0 ? angle + 180 : angle;

			// angle 0
			if ((0 <= angle && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
				q = grad_norm[i][j + 1];
				r = grad_norm[i][j - 1];
			}

			// angle 45
			else if (22.5 <= angle && angle < 67.5) {
				q = grad_norm[i + 1][j - 1];
				r = grad_norm[i - 1][j + 1];
			}	

			// angle 90
			else if (67.5 <= angle && angle < 112.5) {
				q = grad_norm[i + 1][j];
				r = grad_norm[i - 1][j];
			}

			// angle 135
			else if (112.5 <= angle && angle < 157.5) {
				q = grad_norm[i - 1][j - 1];
				r = grad_norm[i + 1][j + 1];
			}

			if (grad_norm[i][j] >= q && grad_norm[i][j] >= r) {
				grad_suppressed[i][j] = grad_norm[i][j];
			}
			else {
				grad_suppressed[i][j] = 0;
			}
		}
	}
}

void hysteresis_thresholding(uchar** grad_suppressed, int height, int width, int low_tr, int high_tr, uchar** grad_hyst) {
	uchar weak = 50, strong = 255;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (grad_suppressed[i][j] >= high_tr) {
				grad_hyst[i][j] = strong;
			}
			else if (grad_suppressed[i][j] < low_tr) {
				grad_hyst[i][j] = 0;
			}
			else {
				grad_hyst[i][j] = weak;
			}
		}
	}

	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			if (grad_hyst[i][j] == weak) {
				if (grad_hyst[i + 1][j - 1] == strong || grad_hyst[i + 1][j] == strong || grad_hyst[i + 1][j + 1] == strong ||
					grad_hyst[i][j - 1] == strong || grad_hyst[i][j + 1] == strong ||
					grad_hyst[i - 1][j - 1] == strong || grad_hyst[i - 1][j] == strong || grad_hyst[i - 1][j + 1] == strong) {
					grad_hyst[i][j] = strong;
				}
				else if (grad_hyst[i + 1][j - 1] == weak || grad_hyst[i + 1][j] == weak || grad_hyst[i + 1][j + 1] == weak ||
					grad_hyst[i][j - 1] == weak || grad_hyst[i][j + 1] == weak ||
					grad_hyst[i - 1][j - 1] == weak || grad_hyst[i - 1][j] == weak || grad_hyst[i - 1][j + 1] == weak) {
					grad_hyst[i][j] = strong;
				}

				else {
					grad_hyst[i][j] = 0;
				}
			}
		}
	}
}

void hysterezis(uchar** input, int height, int width) {
	for (int i = 1; i < height - 1; i++) {
		for (int j = 1; j < width - 1; j++) {
			if (input[i][j] == 50) {
				if (input[i + 1][j - 1] == 255 || input[i + 1][j] == 255 || input[i + 1][j + 1] == 255 ||
					input[i][j - 1] == 255 || input[i][j + 1] == 255 ||
					input[i - 1][j - 1] == 255 || input[i - 1][j] == 255 || input[i - 1][j + 1] == 255) {
					input[i][j] = 255;
				}
				else {
					input[i][j] = 0;
				}
			}
		}
	}
}

void canny_classic(uchar** input, int height, int width, int low_tr, int high_tr, uchar** output) {
	// Compute Gaussian blur
	uchar** input_blurred = new uchar * [height];
	for (int i = 0; i < height; i++) {
		input_blurred[i] = new uchar[width];
	}

	// Compute gradient
	float** grad = new float* [height];
	for (int i = 0; i < height; i++) {
		grad[i] = new float[width];
	}

	float** orientations = new float* [height];
	for (int i = 0; i < height; i++) {
		orientations[i] = new float[width];
	}

	sobel_classic(input, height, width, grad, orientations);
	//fix_borders<float>(grad, height, width, 3);

	// Normalize
	uchar** grad_norm = new uchar * [height];
	for (int i = 0; i < height; i++) {
		grad_norm[i] = new uchar[width];
	}

	normalize_image(grad, height, width, grad_norm);

	// Non-maximum suppression
	uchar** grad_suppressed = new uchar* [height];
	for (int i = 0; i < height; i++) {
		grad_suppressed[i] = new uchar[width];
	}

	non_max_suppression(grad_norm, height, width, orientations, grad_suppressed);

	// Hysteresis thresholding
	hysteresis_thresholding(grad_suppressed, height, width, low_tr, high_tr, output);

	for (int i = 0; i < height; i++) {
		delete[] grad_suppressed[i];
		delete[] grad_norm[i];
		delete[] orientations[i];
		delete[] grad[i];
		delete[] input_blurred[i];
	}

	delete[] grad_suppressed;
	delete[] grad_norm;
	delete[] orientations;
	delete[] grad;
	delete[] input_blurred;
}



