import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv2.imread('kite_runner.jpg', cv2.IMREAD_GRAYSCALE)          # queryImage
img2 = cv2.imread('kite_runner_.jpg', cv2.IMREAD_GRAYSCALE) # trainImage

orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

good = []
good_without_list = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])
        good_without_list.append(m)
# Draw first 10 matches.

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.array([ kp1[m.queryIdx].pt for m in good_without_list ]).reshape(-1,1,2)
    dst_pts = np.array([ kp2[m.trainIdx].pt for m in good_without_list ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.array([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    # print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_without_list, None, flags=2)

plt.imshow(img3),plt.show()
