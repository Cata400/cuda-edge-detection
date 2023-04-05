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

