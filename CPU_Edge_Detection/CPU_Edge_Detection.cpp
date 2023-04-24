#include <iostream>
#include <cstring>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <chrono>

#include "cpu_utils.h"

using namespace cv;

int main()
{
    // Parameters
    bool canny = true;
    int reps = 20;
    std::string image_name, image_path, save_path_opencv, save_path_opencv_gpu, save_path_classic;
    image_name = "Lenna_multiplied_4x4";

    if (canny) {
		image_path = "../Images/" + image_name + ".png";
		save_path_opencv = "../Images/" + image_name + "_canny_opencv.png";
		save_path_opencv_gpu = "../Images/" + image_name + "_canny_opencv_gpu.png";
		save_path_classic = "../Images/" + image_name + "_canny_classic.png";
	}
    else {
		image_path = "../Images/" + image_name + ".png";
		save_path_opencv = "../Images/" + image_name + "_sobel_opencv.png";
		save_path_opencv_gpu = "../Images/" + image_name + "_sobel_opencv_gpu.png";
		save_path_classic = "../Images/" + image_name + "_sobel_classic.png";
	}

    Mat image = imread(image_path, IMREAD_COLOR);

    if (!image.data) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    imshow("Image to detect edges from", image);
    waitKey();
    destroyAllWindows();

    cvtColor(image, image, COLOR_BGR2GRAY);

    // OpenCV edge detection
    {
        Mat edges_opencv;

        auto start = std::chrono::high_resolution_clock::now();

        if (canny) {
            for (int i = 0; i < reps; i++)
                edges_opencv = canny_opencv(image, 80, 160, 3);
        }
        else {
            for (int i = 0; i < reps; i++)
                edges_opencv = sobel_opencv(image);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_time = end - start;

        if (canny) {
            std::cout << "Canny OpenCV took " << elapsed_time / std::chrono::milliseconds(1) << " ms" << std::endl << std::endl;
        }
        else {
            std::cout << "Sobel OpenCV took " << elapsed_time / std::chrono::milliseconds(1) << " ms" << std::endl << std::endl;
        }

        Rect roi(1, 1, image.cols - 2, image.rows - 2);
        edges_opencv = edges_opencv(roi);
        
        imwrite(save_path_opencv, edges_opencv);
    }
    
    // OpenCV GPU edge detection
    {
        cuda::GpuMat image_gpu, edges_opencv_gpu;
        Mat edges_opencv_gpu_recovered;
        image_gpu.upload(image);

        auto start = std::chrono::high_resolution_clock::now();

        if (canny) {
            for (int i = 0; i < reps; i++)
                edges_opencv_gpu = canny_opencv_gpu(image_gpu, 80, 160, 3);
        }
        else {
            for (int i = 0; i < reps; i++)
                edges_opencv_gpu = sobel_opencv_gpu(image_gpu);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_time = end - start;

        if (canny) {
            std::cout << "Canny OpenCV CUDA took " << elapsed_time / std::chrono::milliseconds(1) << " ms" << std::endl << std::endl;
        }
        else {
            std::cout << "Sobel OpenCV CUDA took " << elapsed_time / std::chrono::milliseconds(1) << " ms" << std::endl << std::endl;
        }

        edges_opencv_gpu.download(edges_opencv_gpu_recovered);

        Rect roi(1, 1, image.cols - 2, image.rows - 2);
        edges_opencv_gpu_recovered = edges_opencv_gpu_recovered(roi);

		imwrite(save_path_opencv_gpu, edges_opencv_gpu_recovered);
	}
    
    // Classic Canny
    if (canny) {
        int height = image.rows;
        int width = image.cols;
        uchar* image_data = image.data;

        uchar** image_data_2d = new uchar*[height];
        for (int i = 0; i < height; i++) {
			image_data_2d[i] = &image_data[i * width];
		}

        uchar** image_edges_2d = new uchar *[height];
        for (int i = 0; i < height; i++) {
            image_edges_2d[i] = new uchar[width];
		}

        uchar* image_edges = new uchar[height * width];

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < reps; i++)
            canny_classic(image_data_2d, height, width, 40, 120, image_edges_2d);


        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_time = end - start;

        std::cout << "Canny classic took " << elapsed_time / std::chrono::milliseconds(1) << " ms" << std::endl << std::endl;

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                image_edges[i * width + j] = image_edges_2d[i][j];
			}
		}

        Mat edges_classic = Mat(height, width, image.type(), image_edges);

        Rect roi(1, 1, image.cols - 1, image.rows - 1);
        edges_classic = edges_classic(roi);

        imwrite(save_path_classic, edges_classic);

        for (int i = 0; i < height; i++) {
			delete[] image_edges_2d[i];
		}

        delete[] image_edges_2d;
        delete[] image_edges;
    }
    // Classic Sobel
    else {
        int height = image.rows;
        int width = image.cols;
        uchar* image_data = image.data;

        uchar** image_data_2d = new uchar * [height];
        for (int i = 0; i < height; i++) {
            image_data_2d[i] = &image_data[i * width];
        }

        float** image_edges_2d = new float* [height];
        for (int i = 0; i < height; i++) {
            image_edges_2d[i] = new float[width];
        }

        float* image_edges = new float[height * width];

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < reps; i++)
            sobel_classic(image_data_2d, height, width, image_edges_2d);
 
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_time = end - start;

        std::cout << "Sobel classic took " << elapsed_time / std::chrono::milliseconds(1) << " ms" << std::endl << std::endl;

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                image_edges[i * width + j] = image_edges_2d[i][j];
            }
        }

        Mat edges_classic = Mat(height, width, CV_32FC1, image_edges);

        Rect roi(1, 1, image.cols - 2, image.rows - 2);
        edges_classic = edges_classic(roi);

        imwrite(save_path_classic, edges_classic);

        for (int i = 0; i < height; i++) {
            delete[] image_edges_2d[i];
        }

        delete[] image_edges_2d;
        delete[] image_edges;
    }

    return 0;
}
