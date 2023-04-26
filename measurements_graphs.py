import matplotlib.pyplot as plt


sobel_opencv_time = [555, 845, 1465, 2309, 3932]
sobel_opencv_cuda_time = [478, 618, 748, 923, 1279]
sobel_cpp_time = [555, 1186, 2636, 5477, 9154]
sobel_cuda_v12_time = [7.401, 9.959, 18.445, 83.736, 160.536]
sobel_cuda_v3_time = [27.099, 84.018, 300.179, 507.899, 1333.62]
sobel_cuda_v4_time = [45.711, 156.628, 516.466, 1024.18, 2090.73]

sobel_opencv_acc = [1.092, 1.367, 1.958, 2.501, 3.074]
sobel_v12_acc = [74.99, 119.088, 142.911, 65.408, 57.021]
sobel_v3_acc = [20.48, 14.116, 8.781, 9.160, 6.864]
sobel_v4_acc = [12.141, 7.572, 5.104, 5.348, 4.378]

canny_opencv_time = [223, 585, 598, 747, 1419]
canny_opencv_cuda_time = [100, 257, 221, 397, 598]
canny_cpp_time = [1184, 2278, 4530, 7838, 15585]
canny_cuda_v12_time = [129.789, 183.156, 287.114, 688.238, 1186.35]

canny_opencv_acc = [2.23, 2.276, 2.706, 1.882, 2.373]
canny_v12_acc = [9.122, 12.437, 15.778, 11.389, 13.137]

# Plot Sobel time curves
plt.figure()
plt.plot(sobel_opencv_time, label="OpenCV")
plt.plot(sobel_opencv_cuda_time, label="OpenCV CUDA")
plt.plot(sobel_cpp_time, label="C++")
plt.plot(sobel_cuda_v12_time, label="CUDA V1/V2")
plt.plot(sobel_cuda_v3_time, label="CUDA V3")
plt.plot(sobel_cuda_v4_time, label="CUDA V4")
plt.xticks([0, 1, 2, 3, 4], ["512x512", "512x1024", "1024x1024", "1024x2048", "2048x2048"])
plt.xlabel("Image size")
plt.ylabel("Time [ms]")
plt.legend()
plt.show()

# Plot Sobel acceleration curves
plt.figure()
plt.plot(sobel_opencv_acc, label="OpenCV")
plt.plot(sobel_v12_acc, label="CUDA V1/V2")
plt.plot(sobel_v3_acc, label="CUDA V3")
plt.plot(sobel_v4_acc, label="CUDA V4")
plt.xticks([0, 1, 2, 3, 4], ["512x512", "512x1024", "1024x1024", "1024x2048", "2048x2048"])
plt.xlabel("Image size")
plt.ylabel("Acceleration")
plt.legend()
plt.show()

# Plot Canny time curves
plt.figure()
plt.plot(canny_opencv_time, label="OpenCV")
plt.plot(canny_opencv_cuda_time, label="OpenCV CUDA")
plt.plot(canny_cpp_time, label="C++")
plt.plot(canny_cuda_v12_time, label="CUDA V1/V2")
plt.xticks([0, 1, 2, 3, 4], ["512x512", "512x1024", "1024x1024", "1024x2048", "2048x2048"])
plt.xlabel("Image size")
plt.ylabel("Time [ms]")
plt.legend()
plt.show()

# Plot Canny acceleration curves
plt.figure()
plt.plot(canny_opencv_acc, label="OpenCV")
plt.plot(canny_v12_acc, label="CUDA V1/V2")
plt.xticks([0, 1, 2, 3, 4], ["512x512", "512x1024", "1024x1024", "1024x2048", "2048x2048"])
plt.xlabel("Image size")
plt.ylabel("Acceleration")
plt.legend()
plt.show()
