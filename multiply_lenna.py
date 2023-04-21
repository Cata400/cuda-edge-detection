from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import PIL


image_name = "Lenna"
image_path = "Images/" + image_name + ".png"
repetitions = (4, 4, 1)

image = io.imread(image_path)
image_multiplied = np.tile(image, repetitions)
image_multiplied = PIL.Image.fromarray(image_multiplied)

# plt.figure(), plt.imshow(image_multiplied), plt.title("Multiplied image"), plt.show()
image_multiplied.save(f"Images/{image_name}_multiplied_{repetitions[0]}x{repetitions[1]}.png")