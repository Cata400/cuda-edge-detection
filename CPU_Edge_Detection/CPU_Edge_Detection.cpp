#include <iostream>
#include <cstring>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ctime>
#include <iomanip>

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
        double elapsed_time = double(end - start) / CLOCKS_PER_SEC * 1000;
        std::cout << "OpenCV edge detection took " << std::setprecision(8) << elapsed_time << " ms" << std::endl;

        imwrite(save_path_opencv, edges_opencv);
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
        double elapsed_time = double(end - start) / CLOCKS_PER_SEC * 1000;
        std::cout << "Classic edge detection took " << std::setprecision(8) << elapsed_time << " ms" << std::endl;

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                image_edges[i * width + j] = image_edges_2d[i][j];
			}
		}
        Mat edges_classic = Mat(height, width, CV_32FC1, image_edges);

        imwrite(save_path_classic, edges_classic);

        for (int i = 0; i < height; i++) {
			delete[] image_edges_2d[i];
		}
        delete[] image_edges_2d;
        delete[] image_edges;
    }

    return 0;
 
}
