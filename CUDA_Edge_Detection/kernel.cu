#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <cstring>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

__constant__ float gx[3][3] = { { 1, 0, -1 }, { 2, 0, -2 }, { 1, 0, -1 } };
__constant__ float gy[3][3] = { { 1, 2, 1 },  { 0, 0, 0 }, { -1, -2, -1 } };


__global__ void edge_detect_cuda(uchar* input, float* output, int height, int width, bool pixel) {
    int i, j;

    if (pixel) {
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


int main()
{
    // Parameters
    std::string image_name = "Lenna_multiplied_4x4";
    std::string image_path = "../Images/" + image_name + ".png";
    std::string save_path_cuda = "../Images/" + image_name + "_edge_cuda.png";
    int reps = 100;
    bool pixel = false;

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

    float* image_edges = new float[height * width];
    Mat edges_cuda;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!");
        goto Error;
    }

    // Allocate GPU buffers for input and output image
    uchar* image_data_cuda;
    float* image_edges_cuda;

    cudaStatus = cudaMalloc((void**)&image_data_cuda, height * width * sizeof(uchar));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&image_edges_cuda, height * width * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    
    // Copy input vectors from host memory to GPU buffers
    cudaStatus = cudaMemcpy(image_data_cuda, image_data, height * width * sizeof(uchar), cudaMemcpyHostToDevice); // <---- error here
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    
    // Launch a kernel on the GPU with one thread for each pixel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int i = 0; i < reps; i++) {
        if (pixel) {
            edge_detect_cuda <<< height, width >>> (image_data_cuda, image_edges_cuda, height, width, pixel);
        }
        else {
            edge_detect_cuda <<< grid_size, block_size >>> (image_data_cuda, image_edges_cuda, height, width, pixel);
        }

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
            goto Error;
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
        goto Error;
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
    

Error:
    cudaFree(image_data_cuda);
    cudaFree(image_edges_cuda);

    return cudaStatus;
}

