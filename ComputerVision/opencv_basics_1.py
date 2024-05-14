import cv2 as cv
import numpy as np

print('OpenCV version --> ', cv.__version__)
print('Numpy version --> ', np.__version__)

#Show/Display Images
image_path = 'images/road.jpg'
image = cv.imread(image_path)
print('Shape Of Image is -> ' , image.shape)
# cv.imshow("Road Image", image) # -> image too big

# Rescale Function
def rescaleFrame(frame, scale=0.75) :
  width = int(frame.shape[1] * scale)
  height = int(frame.shape[0] * scale)
  
  return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)

image_resized = rescaleFrame(image, 0.125)
cv.imshow('Road Image Resized', image_resized)

# Grayscale image
gray = cv.cvtColor(image_resized, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Scale Image', gray)

# Blur Image
blur = cv.GaussianBlur(image_resized, (3,3), cv.BORDER_DEFAULT)
cv.imshow('Blurred Image', blur)

# Edge Cascade
canny = cv.Canny(image_resized, 125, 175)
cv.imshow('Edge Cascade', canny)
canny_blurred = cv.Canny(blur, 125, 175) # blurring image improves edge detection
cv.imshow('Edge Cascade with Blurred Image', canny_blurred)

# Other functions
## dilate
## erode
## resize -> that was used above scale function

# Cropping Images
cropped = image_resized[50:100, 250:330]
cv.imshow('Cropped Image',cropped)

#create an empty image of size 500 * 500
blank = np.zeros((500,500, 3), dtype = 'uint8')

# Draw Text on image
cv.putText(blank, 'Hello OpenCV', (100, 100), cv.FONT_HERSHEY_COMPLEX, 1.0, (100, 100, 200, 2))

# Draw on an image
# Rectangles
cv.rectangle(blank, (0,0), (250, 250), (255, 0, 0), thickness=2)
cv.rectangle(blank, (250,250), (500, 500), (0, 100, 0), thickness=cv.FILLED)
cv.rectangle(blank, (0,250), (250, 500), (0, 0, 255), thickness=-1)
# Circle
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 100, (255, 255, 0))
cv.imshow('Draw on Blank', blank)

# Wait for keypress to close the program, as without this it will show
# and close the windows after the program execution finishes
cv.waitKey(0)

# Show Video
# To capture video from webcam
# capture = cv.VideoCapture(0)
# Similarly videos on computer can be shown



