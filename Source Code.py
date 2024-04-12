import cv2
import sys

filename = 'testb2.jpg'

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread(filename)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform face detection
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the result
img_resized = cv2.resize(img, (1000, 600))
cv2.imwrite('sift_output.jpg', img_resized)
cv2.imshow('Face Detection', img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
