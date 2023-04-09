#include <iostream>
#include <cstring>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <ctime>

#include "cpu_utils.h"

using namespace cv;

int main()
{
    std::string image_name = "Lenna";
    std::string image_path = "../Images/" + image_name + ".png";
    std::string save_path_opencv = "../Images/" + image_name + "_edge_opencv.png";
    std::string save_path_opencv_gpu = "../Images/" + image_name + "_edge_opencv_gpu.png";
    std::string save_path_classic = "../Images/" + image_name + "_edge_classic.png";

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

        clock_t start = clock();
        edges_opencv = edge_detect_opencv(image);
        clock_t end = clock();
        double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
        std::cout << "OpenCV edge detection took " << elapsed_time << " seconds" << std::endl << std::endl;

        imwrite(save_path_opencv, edges_opencv);
    }

    // OpenCV GPU edge detection
    {
        cuda::GpuMat image_gpu, edges_opencv_gpu;
        Mat edges_opencv_gpu_recovered;
        image_gpu.upload(image);

		clock_t start = clock();
		edges_opencv_gpu = edge_detect_opencv_gpu(image_gpu);
		clock_t end = clock();
		double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
		std::cout << "OpenCV CUDA edge detection took " << elapsed_time << " seconds" << std::endl; // To Do: check why MAE is so high

        edges_opencv_gpu.download(edges_opencv_gpu_recovered);
		imwrite(save_path_opencv_gpu, edges_opencv_gpu_recovered);

        Mat edges_opencv = edge_detect_opencv(image);
        Mat diff = abs(edges_opencv_gpu_recovered - edges_opencv);
        std::cout << "MAE between OpenCV and OpenCV CUDA: " << cv::sum(diff)[0] / (diff.rows * diff.cols) << std::endl << std::endl;
	}

    // Classic edge detection
    {
        int height = image.rows;
        int width = image.cols;
        uchar* image_data = image.data;

        uchar** image_data_2d = new uchar*[height];
        for (int i = 0; i < height; i++) {
			image_data_2d[i] = &image_data[i * width];
		}

        float** image_edges_2d = new float*[height];
        for (int i = 0; i < height; i++) {
            image_edges_2d[i] = new float[width];
		}

        float* image_edges = new float[height * width];

        clock_t start = clock();
        edge_detect_classic(image_data_2d, height, width, image_edges_2d);
        clock_t end = clock();
        double elapsed_time = double(end - start) / CLOCKS_PER_SEC;
        std::cout << "Classic edge detection took " << elapsed_time << " seconds" << std::endl;

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                image_edges[i * width + j] = image_edges_2d[i][j];
			}
		}

        Mat edges_classic = Mat(height, width, CV_32FC1, image_edges);
        imwrite(save_path_classic, edges_classic);

        Mat edges_opencv = edge_detect_opencv(image);
        Mat diff = abs(edges_classic - edges_opencv);
        std::cout << "MAE between OpenCV and classic edge detection: " << cv::sum(diff)[0] / (diff.rows * diff.cols) << std::endl;

        for (int i = 0; i < height; i++) {
			delete[] image_edges_2d[i];
		}
        delete[] image_edges_2d;
        delete[] image_edges;
    }

    return 0;
 
}
