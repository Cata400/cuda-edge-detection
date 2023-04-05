#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cpu_utils.h"

using namespace cv;

int main()
{
    std::string image_path = "../Images/Lenna.png";
    std::string save_path = "../Images/Lenna_edge.png";

    Mat image = imread(image_path, IMREAD_COLOR);

    if (!image.data) // Check for invalid input
    {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    cvtColor(image, image, COLOR_BGR2GRAY);

    imshow("Image", image);
    waitKey(0);
    destroyAllWindows();

    imwrite(save_path, image);
    return 0;
}
