import cv2
import os
import time

img1 = cv2.imread("hpatches/i_ajuntament/1.ppm", 0)
img2 = cv2.imread("hpatches/i_ajuntament/2.ppm", 0)

surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)

start = time.time()

kp1, des1 = surf.detectAndCompute(img1, None)
kp2, des2 = surf.detectAndCompute(img2, None)

bf = cv2.BFMatcher()

matches = bf.knnMatch(des1, des2, k=2)

good = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

end = time.time()

result = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    good,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

os.makedirs("results", exist_ok=True)

cv2.imwrite("results/surf_matches.png", result)

print("=" * 40)
print("SURF RESULTS")
print("=" * 40)
print("Image1 keypoints :", len(kp1))
print("Image2 keypoints :", len(kp2))
print("Good Matches     :", len(good))
print("Execution Time   :", end - start)
print("Saved : results/surf_matches.png")
