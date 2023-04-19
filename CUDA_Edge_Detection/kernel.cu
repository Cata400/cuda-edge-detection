
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <cstring>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;


__global__ void edge_detect_cuda(uchar* input, float* output, int height, int width) {
    //int i = threadIdx.x + blockIdx.x * blockDim.x;
    //int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    int i = blockIdx.x;
    int j = threadIdx.x;

    if (i > 0 && i < height - 1 && j > 0 && j < width - 1) {
        float gx = input[(i - 1) * width + (j - 1)] + 2 * input[i * width + (j - 1)] + input[(i + 1) * width + (j - 1)] - input[(i - 1) * width + (j + 1)] - 2 * input[i * width + (j + 1)] - input[(i + 1) * width + (j + 1)];
		float gy = input[(i - 1) * width + (j - 1)] + 2 * input[(i - 1) * width + j] + input[(i - 1) * width + (j + 1)] - input[(i + 1) * width + (j - 1)] - 2 * input[(i + 1) * width + j] - input[(i + 1) * width + (j + 1)];
		output[i * width + j] = sqrt(gx * gx + gy * gy);
	}
    else {
		output[i * width + j] = 0;
	}
}

int main()
{
    std::string image_name = "Lenna";
    std::string image_path = "../Images/" + image_name + ".png";
    std::string save_path_cuda = "../Images/" + image_name + "_edge_cuda.png";

    // Read image
    Mat image = imread(image_path, IMREAD_COLOR);

    int height = image.rows;
    int width = image.cols;

    dim3 block(height / 4 , width / 4);
    dim3 grid(4, 4);

    if (!image.data) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    imshow("Image to detect edges from", image);
    waitKey();
    destroyAllWindows();

    cvtColor(image, image, COLOR_BGR2GRAY);

    uchar* image_data = image.data;

    float** image_edges_2d = new float*[height];
    image_edges_2d[0] = new float[height * width];

    for (int i = 1; i < height; i++) {
		image_edges_2d[i] = image_edges_2d[i - 1] + width;
	}

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for input and output image
    uchar* image_data_cuda;
    float* image_edges_2d_cuda;

    cudaStatus = cudaMalloc((void**)&image_data_cuda, height * width * sizeof(uchar));
    if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

    cudaStatus = cudaMalloc((void**)&image_edges_2d_cuda, height * width * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers
    cudaStatus = cudaMemcpy(image_data_cuda, image_data, height * width * sizeof(uchar), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

    
    // Launch a kernel on the GPU with one thread for each pixel
    edge_detect_cuda<<<height, width >>>(image_data_cuda, image_edges_2d_cuda, height, width);

    
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
    
    
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(image_edges_2d[0], image_edges_2d_cuda, height * width * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    /*
    Mat edges_cuda = Mat(height, width, CV_32F, image_edges_2d[0]);
    imwrite(save_path_cuda, edges_cuda);

    /*
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    for (int i = 0; i < height; i++) {
        delete[] image_edges_2d[i];
    }
    delete[] image_edges_2d;
    */

Error:
    cudaFree(image_data_cuda);
    cudaFree(image_edges_2d_cuda);

    return cudaStatus;
}

