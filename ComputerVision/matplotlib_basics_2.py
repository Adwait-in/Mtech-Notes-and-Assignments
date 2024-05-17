import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load Image
beach_image = cv.imread('images/beach.jpg')

# Resized image
resized_beach_image = cv.resize(beach_image, (400, 400), interpolation=cv.INTER_AREA)
cv.imshow('Resized Image', resized_beach_image)

grayscaled_beach_image = cv.cvtColor(resized_beach_image, cv.COLOR_BGR2GRAY)

grayscaled_beach_image_manual = np.dot(resized_beach_image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
print('Image shape -> ', grayscaled_beach_image_manual.shape)

# Show images in the same line with matplotlib
plt.figure(figsize=(10, 6)) # figsize is in inches
plt.subplot(1, 3, 1) # rows, columns, index
plt.imshow(resized_beach_image)
plt.title('Resized Original')
plt.subplot(1, 3, 2) # rows, columns, index
plt.imshow(grayscaled_beach_image, cmap='gray')
plt.title('Gray Scale Original')
plt.subplot(1, 3, 3) # rows, columns, index
plt.imshow(grayscaled_beach_image_manual, cmap='gray')
plt.title('Manual Gray Scale Original')

plt.show()