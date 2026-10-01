import cv2
import os
import time

img1 = cv2.imread("hpatches/i_ajuntament/1.ppm", 0)
img2 = cv2.imread("hpatches/i_ajuntament/2.ppm", 0)

orb = cv2.ORB_create()

kp1 = orb.detect(img1, None)
kp2 = orb.detect(img2, None)

latch = cv2.xfeatures2d.LATCH_create()

start = time.time()

kp1, des1 = latch.compute(img1, kp1)
kp2, des2 = latch.compute(img2, kp2)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

end = time.time()

result = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    matches[:100],
    None,
    flags=2
)

os.makedirs("results", exist_ok=True)

cv2.imwrite("results/latch_matches.png", result)

print("=" * 40)
print("LATCH RESULTS")
print("=" * 40)
print("Image1 keypoints :", len(kp1))
print("Image2 keypoints :", len(kp2))
print("Matches          :", len(matches))
print("Execution Time   :", end - start)
print("Saved : results/latch_matches.png")
