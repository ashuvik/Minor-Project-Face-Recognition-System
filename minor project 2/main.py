
import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt

img1=cv2.imread('image1.jpg')

#img1=cv2.imread('image2.webp')

#img1=cv2.imread('image3.jpg')

plt.imshow(img1[:,:,::-1])

plt.show()

result = DeepFace.analyze(img1, actions = ['emotion'])

print(result)