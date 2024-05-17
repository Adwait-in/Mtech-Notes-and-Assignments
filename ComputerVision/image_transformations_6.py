import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

road_image = cv.imread('images/road.jpg')
road_resized = cv.resize(road_image, 
                         [int(road_image.shape[0] * 0.1), int(road_image.shape[0] * 0.1)], 
                         interpolation=cv.INTER_AREA)

# Translate Image
# consider y as rows and x as columns in the pixel matrix
def translate(image, dx, dy):
  translatedImage = np.zeros_like(image)
  for y in range(image.shape[0]):
    for x in range(image.shape[1]):
      x_trans = x + dx
      y_trans = y + dy
      
      #check the translated coordinates are in between bounds of the translate image array
      if 0<= x_trans < image.shape[1] and 0 <= y_trans < image.shape[0] :
        translatedImage[y_trans, x_trans] = image[y, x]
      
  return translatedImage
        
transplatedImage = translate(road_resized, 80, 120)

# Rotate Image
# rotation matrix
# |x| = | cos(theta)  -sin(theta) | |u|
# |y|   | sin(theta)   cos(theta) | |v|
def rotate(image, angleInRadians):
  rotatedImage = np.zeros_like(image)
  height, width, channel = image.shape
  cos_theta = np.cos(angleInRadians)
  sin_theta = np.sin(angleInRadians)
  for y in range(height):
    for x in range(width):
      #translate pixel to center
      x_trans = x - width/2
      y_trans = y - height/2
      #rotate coordinates by angle - > add center distances to remove translation
      x_rotated = int(x_trans*cos_theta - y_trans*sin_theta + width/2)
      y_rotated = int(x_trans*sin_theta + y_trans*cos_theta + height/2)
      #check if pixel is in the image bounds
      if 0 <= x_rotated < width and 0 <= y_rotated < height:
        rotatedImage[y_rotated, x_rotated] = image[y,x]
  return rotatedImage

rotatedImage = rotate(road_resized, np.radians(-30))

# Resize
def resize(image, xFactor, yFactor):
  height, width, channels = image.shape
  scaledHight = int(yFactor * height)
  scaledWidth = int(xFactor * width)
  resized = np.zeros((scaledHight, scaledWidth, channels), dtype=np.uint8)
  for y in range(scaledHight):
    for x in range(scaledWidth):
      scaledX = int(x/xFactor)
      scaledY = int(y/yFactor)
      if 0 <= scaledX < width and 0 <= scaledY < height:
        resized[y,x] = image[scaledY, scaledX]
  return resized

resized_image_manual = resize(road_resized, 0.5, 0.5)
print('Original Shape -> ' ,  road_resized.shape)
print('Resized Shape -> ' , resized_image_manual.shape)
# Resized images dont show in the shape as expected in plot so not shown

# Shear
def shear(image, factor, type='x'):
  height, width, channel = image.shape
  if type=='x':
    shearedWidth = width + int(factor * height)
    shearedHeight = height
  else :
    shearedWidth = width
    shearedHeight = height + int(factor * width)
  shearedImage = np.zeros((shearedHeight, shearedWidth, channel), dtype=np.uint8)
  for y in range(height):
    for x in range(width):
      shearX = x + int(factor * y)
      shearY = y + int(factor * x)
      if 0 <= shearX < shearedWidth and 0 <= shearY < shearedHeight :
        shearedImage[shearY, shearX] = image[y, x]
  return shearedImage

shearedImage = shear(road_resized, 0.5)
shearedImageY = shear(road_resized, 0.5, 'y')

# Show Transformed Images
plt.figure(figsize=(12, 4))

plt.subplot(1,4,1)
plt.title('Original')
plt.imshow(road_resized)
plt.axis('off')
plt.subplot(1,4,2)
plt.title('Translated 80, 120')
plt.imshow(transplatedImage)
plt.axis('off')
plt.subplot(1,4,3)
plt.title('Rotated 30deg')
plt.imshow(rotatedImage)
plt.axis('off')
plt.subplot(1,4,4)
plt.title('Sheared 0.5')
plt.imshow(shearedImage)
plt.axis('off')

plt.tight_layout()
plt.show()