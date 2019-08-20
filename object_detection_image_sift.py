import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('./images/kite_runner.jpg', cv2.IMREAD_GRAYSCALE)          # queryImage
img2 = cv2.imread('./images/kite_runner_.jpg', cv2.IMREAD_GRAYSCALE) # trainImage

orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# Draw first 10 matches.
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

cv2.imshow('image',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
