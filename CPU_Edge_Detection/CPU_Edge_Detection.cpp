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
    std::string image_path = "../Images/Lenna.png";
    std::string save_path_opencv = "../Images/Lenna_edge_opencv.png";
    std::string save_path_opencv_gpu = "../Images/Lenna_edge_opencv_gpu.png";
    std::string save_path_classic = "../Images/Lenna_edge_classic.png";

    Mat image = imread(image_path, IMREAD_COLOR);

    if (!image.data) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    cvtColor(image, image, COLOR_BGR2GRAY);

    // OpenCV edge detection
    {
        clock_t start = clock();
        Mat edges_opencv = edge_detect_opencv(image);
        clock_t end = clock();
        double elapsed_time = double(end - start) / CLOCKS_PER_SEC * 1000;
        std::cout << "OpenCV edge detection took " << std::setprecision(8) << elapsed_time << " ms" << std::endl;
        imwrite(save_path_opencv, edges_opencv);
    }

    return 0;
 
}
