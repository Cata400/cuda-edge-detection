from skimage import io
import numpy as np
import matplotlib.pyplot as plt

image_name = "Lenna_multiplied_4x4"
image_path = "Images/" + image_name + ".png"
save_path_opencv = "Images/" + image_name + "_edge_opencv.png"
save_path_opencv_gpu = "Images/" + image_name + "_edge_opencv_gpu.png"
save_path_classic = "Images/" + image_name + "_edge_classic.png"
save_path_cuda = "Images/" + image_name + "_edge_cuda.png"

image_opencv = io.imread(save_path_opencv)
image_classic = io.imread(save_path_classic)
image_opencv_gpu = io.imread(save_path_opencv_gpu)
image_cuda = io.imread(save_path_cuda)

print(f"Image OpenCV -> shape: {image_opencv.shape}, dtype: {image_opencv.dtype}, min: {np.min(image_opencv)}, max: {np.max(image_opencv)}\n")
print(f"Image Classic -> shape: {image_classic.shape}, dtype: {image_classic.dtype}, min: {np.min(image_classic)}, max: {np.max(image_classic)}")
print(f"Difference -> min: {np.min(image_opencv - image_classic)}, max: {np.max(image_opencv - image_classic)}")
print(f"MAE: {np.mean(np.abs(image_opencv - image_classic))}\n")

print(f"Image OpenCV CUDA -> shape: {image_opencv_gpu.shape}, dtype: {image_opencv_gpu.dtype}, min: {np.min(image_opencv_gpu)}, max: {np.max(image_opencv_gpu)}")
print(f"Difference -> min: {np.min(image_opencv - image_opencv_gpu)}, max: {np.max(image_opencv - image_opencv_gpu)}")
print(f"MAE: {np.mean(np.abs(image_opencv - image_opencv_gpu))}\n")

print(f"Image CUDA -> shape: {image_cuda.shape}, dtype: {image_cuda.dtype}, min: {np.min(image_cuda)}, max: {np.max(image_cuda)}")
print(f"Difference -> min: {np.min(image_opencv - image_cuda)}, max: {np.max(image_opencv - image_cuda)}")
print(f"MAE: {np.mean(np.abs(image_opencv - image_cuda))}\n")