import cv2
import os
import time

img1 = cv2.imread("hpatches/i_ajuntament/1.ppm", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("hpatches/i_ajuntament/2.ppm", cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()

start = time.time()

# Detect keypoints and descriptors
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Brute Force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# KNN matching
raw_matches = bf.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good_matches = []

for m, n in raw_matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# RANSAC
inlier_matches = []

if len(good_matches) >= 4:

    src_pts = []
    dst_pts = []

    for m in good_matches:
        src_pts.append(kp1[m.queryIdx].pt)
        dst_pts.append(kp2[m.trainIdx].pt)

    src_pts = __import__("numpy").float32(src_pts).reshape(-1, 1, 2)
    dst_pts = __import__("numpy").float32(dst_pts).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        5.0
    )

    if mask is not None:
        for i, m in enumerate(good_matches):
            if mask[i]:
                inlier_matches.append(m)

end = time.time()

# Draw only RANSAC inliers
result = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    inlier_matches,
    None,
    flags=2
)

os.makedirs("results", exist_ok=True)

cv2.imwrite(
    "results/orb_ransac_matches.png",
    result
)

print("=" * 40)
print("ORB BASELINE RESULTS")
print("=" * 40)

print("Image1 keypoints :", len(kp1))
print("Image2 keypoints :", len(kp2))
print("Good matches     :", len(good_matches))
print("RANSAC inliers    :", len(inlier_matches))
print("Execution Time    :", end - start)

print("Saved : results/orb_ransac_matches.png")
