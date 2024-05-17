import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# load image as grayscale
beach_image = cv.imread('images/beach.jpg', cv.IMREAD_GRAYSCALE)
print('Grayscale beach image shape', beach_image.shape)

# Calculate histogram Manually

# initialize histogram bin
hist_bin = np.zeros(256) 
# create array of lenght 256 as there are 256 different levels of intensity

# calculate histogram manually
for i in range(beach_image.shape[0]) :
  for j in range(beach_image.shape[1]):
    hist_bin[beach_image[i,j]] += 1
# print(hist_bin)

# Calculate histogram with inbuilt function
hist_bin_inbuilt = cv.calcHist([beach_image], [0], None, [256], [0,256])

# plot histogram
plt.figure(figsize=(12, 6))

plt.subplot(1,2,1)
plt.plot(hist_bin, color='black')
plt.title('Manual Histogram')
plt.xlabel('pixel values')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(hist_bin_inbuilt, color='blue')
plt.title('Inbuilt Function Histogram')
plt.xlabel('pixel values')
plt.ylabel('Frequency')
plt.grid(True)

plt.show()