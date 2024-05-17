import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

beach_image = cv.imread('images/beach.jpg', cv.IMREAD_GRAYSCALE)

beach_image_resized = cv.resize(beach_image,(int(beach_image.shape[0] * 0.1), int(beach_image.shape[1] * 0.1)), interpolation=cv.INTER_AREA)

def manual_histogram_equilization(image):
  hist, _ = np.histogram(image.flatten(), 256, [0, 256])
  cdf = hist.cumsum() # cumulutive Distribution function
  cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
  cdf_normalized = cdf_normalized.astype('uint8')
  equalized_image = cdf_normalized[image]
  return equalized_image

equalized_image_manual = manual_histogram_equilization(beach_image_resized)
equalized_image_inbuilt = cv.equalizeHist(beach_image_resized)

plt.figure(figsize=(12, 4))

plt.subplot(1,3,1)
plt.imshow(beach_image_resized, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(equalized_image_manual, cmap='gray')
plt.title('Manual Equalized Image')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(equalized_image_inbuilt, cmap='gray')
plt.title('Inbuilt Equalized Image')
plt.axis('off')

plt.tight_layout()
plt.show()


