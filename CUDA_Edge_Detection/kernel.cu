#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#include <stdio.h>
#include <iostream>
#include <cstring>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

#define PI 3.14159265358979323846

__constant__ float gx[3][3] = { { 1, 0, -1 }, { 2, 0, -2 }, { 1, 0, -1 } };
__constant__ float gy[3][3] = { { 1, 2, 1 },  { 0, 0, 0 }, { -1, -2, -1 } };

// Calculate the sobel filter output for a single pixel
__global__ void sobel_pixel(uchar* input, float* output, int height, int width, bool small);

// Calculate the sobel filter output for every pixel in a 32 x 32 region
__global__ void sobel_region(uchar* input, float* output, int height, int width, const int block_x, const int block_y);

// Calculate the sobel filter output for every pixel in a 32 x 32 region
__global__ void sobel_region_shared(uchar* input, float* output, int height, int width, const int block_x, const int block_y);

// Calculate the canny filter output for a single pixel
__global__ void canny_pixel_1 (uchar* input, float* grad, float* orientations, int height, int width, bool small);

void normalize(float* input, uchar* output, int height, int width);

__global__ void canny_pixel_2(uchar* grad_norm, float* orientations, uchar* grad_suppressed, uchar* output, int height, int width, int low_tr, int high_tr, bool small);


int main() {
    // Parameters
    bool canny = true;
    int reps = 50;
    bool region = false;
    bool shared = true;
    std::string image_name, image_path, save_path_cuda;
    image_name = "Lenna_multiplied_4x4";


    if (canny) {
		image_path = "../Images/" + image_name + ".png";
		save_path_cuda = "../Images/" + image_name + "_canny_cuda.png";
	}
    else {
		image_path = "../Images/" + image_name + ".png";
		save_path_cuda = "../Images/" + image_name + "_sobel_cuda.png";
	}   

    // Read image
    Mat image = imread(image_path, IMREAD_COLOR);

    if (!image.data) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    imshow("Image to detect edges from", image);
    waitKey();
    destroyAllWindows();

    cvtColor(image, image, COLOR_BGR2GRAY);

    // Allocate arrays and dimensions
    int height = image.rows;
    int width = image.cols;

    int block_size_x = 8;
    int block_size_y = 8;
    dim3 block_size(block_size_x, block_size_y);

    int region_size_x = (height + 32 - 1) / 32;
    int region_size_y = (width + 32 - 1) / 32;

    dim3 region_size(region_size_x, region_size_y);

    int grid_size_x = (height + block_size_x - 1) / block_size_x;
    int grid_size_y = (width + block_size_y - 1) / block_size_y;
    dim3 grid_size(grid_size_x, grid_size_y);

    uchar* image_data = new uchar[height * width];
    int index = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            image_data[index++] = image.at<uchar>(i, j);
        }
    }

    cudaError_t cudaStatus;

    // Canny filter
    if (canny) {
        uchar* image_edges = new uchar[height * width];
        Mat edges_cuda;

        // Choose which GPU to run on, change this on a multi-GPU system
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!");
            goto Error_canny;
        }

        // Allocate GPU buffers for input and output image
        uchar* image_data_cuda;
        uchar* image_edges_cuda;
        float* grad;
        float* orientations;
        uchar* grad_norm;
        uchar* grad_suppressed;

        float* grad_cpu = new float[height * width];
        uchar* grad_norm_cpu = new uchar[height * width];

        cudaStatus = cudaMalloc((void**)&image_data_cuda, height * width * sizeof(uchar));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error_canny;
        }

        cudaStatus = cudaMalloc((void**)&image_edges_cuda, height * width * sizeof(uchar));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error_canny;
        }

        // Copy input vectors from host memory to GPU buffers
        cudaStatus = cudaMemcpy(image_data_cuda, image_data, height * width * sizeof(uchar), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error_canny;
        }

        // Launch a kernel on the GPU with one thread for each pixel
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        for (int i = 0; i < reps; i++) {
            cudaStatus = cudaMalloc((void**)&grad, height * width * sizeof(float));
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMalloc failed!");
                goto Error_canny;
            }

            cudaStatus = cudaMalloc((void**)&orientations, height * width * sizeof(float));
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMalloc failed!");
                goto Error_canny;
            }

            cudaStatus = cudaMalloc((void**)&grad_norm, height * width * sizeof(uchar));
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMalloc failed!");
                goto Error_canny;
            }

            cudaStatus = cudaMalloc((void**)&grad_suppressed, height * width * sizeof(uchar));
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMalloc failed!");
                goto Error_canny;
            }

            // Compute the gradient and orientation
            if (width <= 1024) {
                canny_pixel_1 <<< height, width >>> (image_data_cuda, grad, orientations, height, width, true);
            }
            else {
                canny_pixel_1 <<< grid_size, block_size >>> (image_data_cuda, grad, orientations, height, width, false);
            }

            // Copy grad from GPU buffer to host memory
            cudaStatus = cudaMemcpy(grad_cpu, grad, height * width * sizeof(float), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
				goto Error_canny;
			}

            // Compute the gradient norm
            normalize(grad_cpu, grad_norm_cpu, height, width);

            // Copy grad_norm from host memory to GPU buffer
            cudaStatus = cudaMemcpy(grad_norm, grad_norm_cpu, height * width * sizeof(uchar), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
				goto Error_canny;
			}

            // Non-maximum suppression and hysteresis
            if (width <= 1024) {
                canny_pixel_2 <<< height, width >> > (grad_norm, orientations, grad_suppressed, image_edges_cuda, height, width, 40, 120, true);
            }
            else {
                canny_pixel_2 <<< grid_size, block_size >> > (grad_norm, orientations, grad_suppressed, image_edges_cuda, height, width, 40, 120, false);
            }


            // Check for any errors launching the kernel
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
                goto Error_canny;
            }

            cudaFree(grad_suppressed);
            cudaFree(grad_norm);
            cudaFree(orientations);
            cudaFree(grad);

            // cudaDeviceSynchronize waits for the kernel to finish, and returns
            // any errors encountered during the launch.
            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
                goto Error_canny;
            }
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << "Canny CUDA took " << elapsedTime << " ms" << std::endl;

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(image_edges, image_edges_cuda, height * width * sizeof(uchar), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error_canny;
        }

        // Save image
        edges_cuda = Mat(height, width, image.type(), image_edges);
        Rect roi(1, 1, width - 2, height - 2);
        edges_cuda = edges_cuda(roi);
        imwrite(save_path_cuda, edges_cuda);


        // cudaDeviceReset must be called before exiting in order for profiling and
        // tracing tools such as Nsight and Visual Profiler to show complete traces.
        cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceReset failed!");
            return 1;
        }

        delete[] image_edges;
        delete[] image_data;



    Error_canny:
        cudaFree(image_data_cuda);
        cudaFree(image_edges_cuda);

        return cudaStatus;
    }
    
    // Sobel filter
    else {
        float* image_edges = new float[height * width];
        Mat edges_cuda;

        // Choose which GPU to run on, change this on a multi-GPU system
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!");
            goto Error_sobel;
        }

        // Allocate GPU buffers for input and output image
        uchar* image_data_cuda;
        float* image_edges_cuda;

        cudaStatus = cudaMalloc((void**)&image_data_cuda, height * width * sizeof(uchar));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error_sobel;
        }

        cudaStatus = cudaMalloc((void**)&image_edges_cuda, height * width * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error_sobel;
        }

        // Copy input vectors from host memory to GPU buffers
        cudaStatus = cudaMemcpy(image_data_cuda, image_data, height * width * sizeof(uchar), cudaMemcpyHostToDevice); // <---- error here
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error_sobel;
        }

        // Launch a kernel on the GPU with one thread for each pixel
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        for (int i = 0; i < reps; i++) {
            if (region) {
                if (shared) {
                    sobel_region_shared <<< region_size, 1 >>> (image_data_cuda, image_edges_cuda, height, width, region_size_x, region_size_y);
                }
                else {
                    sobel_region <<< region_size, 1 >>> (image_data_cuda, image_edges_cuda, height, width, region_size_x, region_size_y);
                }
            }
            else if (width <= 1024) {
                sobel_pixel <<< height, width >>> (image_data_cuda, image_edges_cuda, height, width, true);
            }
            else {
                sobel_pixel <<< grid_size, block_size >>> (image_data_cuda, image_edges_cuda, height, width, false);
            }

            // Check for any errors launching the kernel
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
                goto Error_sobel;
            }

            // cudaDeviceSynchronize waits for the kernel to finish, and returns
            // any errors encountered during the launch.
            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
                goto Error_sobel;
            }
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << "CUDA edge detection took " << elapsedTime << " ms" << std::endl;

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(image_edges, image_edges_cuda, height * width * sizeof(float), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error_sobel;
        }

        // Save image
        edges_cuda = Mat(height, width, CV_32FC1, image_edges);
        Rect roi(1, 1, width - 2, height - 2);
        edges_cuda = edges_cuda(roi);
        imwrite(save_path_cuda, edges_cuda);


        // cudaDeviceReset must be called before exiting in order for profiling and
        // tracing tools such as Nsight and Visual Profiler to show complete traces.
        cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceReset failed!");
            return 1;
        }

        delete[] image_edges;
        delete[] image_data;



    Error_sobel:
        cudaFree(image_data_cuda);
        cudaFree(image_edges_cuda);

        return cudaStatus;
    }
    
}


__global__ void sobel_pixel(uchar* input, float* output, int height, int width, bool small) {
    int i, j;

    if (small) {
        i = blockIdx.x;
        j = threadIdx.x;
    }
    else {
        i = blockIdx.x * blockDim.x + threadIdx.x;
        j = blockIdx.y * blockDim.y + threadIdx.y;
    }

    float grad_x = 0, grad_y = 0;

    if (i > 0 && i < height - 1 && j > 0 && j < width - 1) {
        for (int k = -1; k < 2; k++) {
            for (int l = -1; l < 2; l++) {
                grad_x += input[(i + k) * width + (j + l)] * gx[k + 1][l + 1];
                grad_y += input[(i + k) * width + (j + l)] * gy[k + 1][l + 1];
            }
        }
        output[i * width + j] = sqrt(grad_x * grad_x + grad_y * grad_y);
    }
    else {
        output[i * width + j] = 0;
    }
}

__global__ void sobel_region(uchar* input, float* output, int height, int width, const int block_x, const int block_y) {
    int x_begin = blockIdx.x * block_x;
    int y_begin = blockIdx.y * block_y;
    int x_end = x_begin + block_x;
    int y_end = y_begin + block_y;

    for (int i = x_begin; i < x_end; i++) {
        for (int j = y_begin; j < y_end; j++) {
            float grad_x = 0, grad_y = 0;

            if (i * width + j < height * width && j < width) {
                if (i > 0 && i < height - 1 && j > 0 && j < width - 1) {
                    for (int k = -1; k < 2; k++) {
                        for (int l = -1; l < 2; l++) {
                            grad_x += input[(i + k) * width + (j + l)] * gx[k + 1][l + 1];
                            grad_y += input[(i + k) * width + (j + l)] * gy[k + 1][l + 1];
                        }
                    }
                    output[i * width + j] = sqrt(grad_x * grad_x + grad_y * grad_y);
                }
                else {
                    output[i * width + j] = 0;
                }
            }
        }
    }
}

__global__ void sobel_region_shared(uchar* input, float* output, int height, int width, const int block_x, const int block_y) {
    int x_begin = blockIdx.x * block_x;
    int y_begin = blockIdx.y * block_y;
    int x_end = x_begin + block_x;
    if (x_end > height) {
        x_end = height;
    }
    int y_end = y_begin + block_y;
    if (y_end > width) {
        y_end = width;
    }

    x_begin -= 1;
    x_end += 1;
    y_begin -= 1;
    y_end += 1;
    __shared__ uchar input_copy[100][100];
    for (int i = 0; i < block_x + 2; i++) {
        for (int j = 0; j < block_y + 2; j++) {
            if ((i + x_begin >= 0) && (i + x_begin < height) && (j + y_begin >= 0) && (j + y_begin < width)) {
                input_copy[i][j] = input[(i + x_begin) * width + (j + y_begin)];
            }
            else {
                input_copy[i][j] = 0;
            }
        }
    }

    __syncthreads();

    for (int i = x_begin + 1; i < x_end - 1; i++) {
        for (int j = y_begin + 1; j < y_end - 1; j++) {
            float grad_x = 0, grad_y = 0;

            if (i > 0 && i < height - 1 && j > 0 && j < width - 1) {
                for (int k = -1; k < 2; k++) {
                    for (int l = -1; l < 2; l++) {
                        grad_x += input_copy[i - x_begin + k][j - y_begin + l] * gx[k + 1][l + 1];
                        grad_y += input_copy[i - x_begin + k][j - y_begin + l] * gy[k + 1][l + 1];
                    }
                }
                output[i * width + j] = sqrt(grad_x * grad_x + grad_y * grad_y);
            }
            else {
                output[i * width + j] = 0;
            }
        }
    }
}

__global__ void canny_pixel_1(uchar* input, float* grad, float* orientations, int height, int width, bool small) {
    int i, j;

    if (small) {
        i = blockIdx.x;
        j = threadIdx.x;
    }
    else {
        i = blockIdx.x * blockDim.x + threadIdx.x;
        j = blockIdx.y * blockDim.y + threadIdx.y;
    }

    float grad_x = 0, grad_y = 0;

    // Compute gradient and orientation
    if (i > 0 && i < height - 1 && j > 0 && j < width - 1) {
        for (int k = -1; k < 2; k++) {
            for (int l = -1; l < 2; l++) {
                grad_x += input[(i + k) * width + (j + l)] * gx[k + 1][l + 1];
                grad_y += input[(i + k) * width + (j + l)] * gy[k + 1][l + 1];
            }
        }
        grad[i * width + j] = sqrt(grad_x * grad_x + grad_y * grad_y);
        orientations[i * width + j] = atan2(grad_y, grad_x);
    }
    else {
        grad[i * width + j] = 0;
    }
}

void normalize(float* input, uchar* output, int height, int width) {
	float max = 0, min = 255;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            max = max < input[i * width + j] ? input[i * width + j] : max;
            min = min > input[i * width + j] ? input[i * width + j] : min;
		}
	}

    float range = max - min;
    float scale = 255 / range;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
			output[i * width + j] = (uchar)(input[i * width + j] * scale);
		}
	}
}

__global__ void canny_pixel_2(uchar* grad_norm, float* orientations, uchar* grad_suppressed, uchar* output, int height, int width, int low_tr, int high_tr, bool small) {
    int i, j;

    if (small) {
        i = blockIdx.x;
        j = threadIdx.x;
    }
    else {
        i = blockIdx.x * blockDim.x + threadIdx.x;
        j = blockIdx.y * blockDim.y + threadIdx.y;
    }

    if (i > 0 && i < height - 1 && j > 0 && j < width - 1) {
        // Non-maximum suppression
        float angle;
        int q = 255, r = 255;
        angle = orientations[i * width + j] * 180 / PI;
        angle = angle < 0 ? angle + 180 : angle;

        // angle 0
        if ((0 <= angle && angle < 22.5) || (157.5 <= angle && angle <= 180)) {
            q = grad_norm[i * width + (j + 1)];
            r = grad_norm[i * width + (j - 1)];
        }

        // angle 45
        else if (22.5 <= angle && angle < 67.5) {
            q = grad_norm[(i + 1) * width + (j - 1)];
            r = grad_norm[(i - 1) * width + (j + 1)];
        }

        // angle 90
        else if (67.5 <= angle && angle < 112.5) {
            q = grad_norm[(i + 1) * width + j];
            r = grad_norm[(i - 1) * width + j];
        }

        // angle 135
        else if (112.5 <= angle && angle < 157.5) {
            q = grad_norm[(i + 1) * width + (j + 1)];
            r = grad_norm[(i - 1) * width + (j - 1)];
        }

        if (grad_norm[i * width + j] >= q && grad_norm[i * width + j] >= r) {
            grad_suppressed[i * width + j] = grad_norm[i * width + j];
        }
        else {
            grad_suppressed[i * width + j] = 0;
        }
    }

    __syncthreads();

    if (i > 0 && i < height - 1 && j > 0 && j < width - 1) {
        // Thresholding
        int weak = 50, strong = 255;

        if (grad_suppressed[i * width + j] >= high_tr) {
            output[i * width + j] = strong;
        }
        else if (grad_suppressed[i * width + j] < low_tr) {
            output[i * width + j] = 0;
        }
        else {
            output[i * width + j] = weak;
        }

        __syncthreads();

        // Hysteresis
        if (output[i * width + j] == weak) {
            if (output[(i + 1) * width + j] == strong || output[(i - 1) * width + j] == strong || output[i * width + (j + 1)] == strong || output[i * width + (j - 1)] == strong || output[(i + 1) * width + (j + 1)] == strong || output[(i + 1) * width + (j - 1)] == strong || output[(i - 1) * width + (j + 1)] == strong || output[(i - 1) * width + (j - 1)] == strong) {
                output[i * width + j] = strong;
            }
            else if (output[(i + 1) * width + j] == weak || output[(i - 1) * width + j] == weak || output[i * width + (j + 1)] == weak || output[i * width + (j - 1)] == weak || output[(i + 1) * width + (j + 1)] == weak || output[(i + 1) * width + (j - 1)] == weak || output[(i - 1) * width + (j + 1)] == weak || output[(i - 1) * width + (j - 1)] == weak) {
                output[i * width + j] = strong;
            }
        }
    }

    else {
        output[i * width + j] = 0;
    }
}