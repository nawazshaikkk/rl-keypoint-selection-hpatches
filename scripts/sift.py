import cv2
import os
import time

# Dataset path
img1_path = "hpatches/i_ajuntament/1.ppm"
img2_path = "hpatches/i_ajuntament/2.ppm"

# Read images
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Error: Images not found!")
    exit()

# Create SIFT detector
sift = cv2.SIFT_create()

start = time.time()

# Detect keypoints and descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

end = time.time()

# Match descriptors
bf = cv2.BFMatcher()

matches = bf.knnMatch(des1, des2, k=2)

good = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

# Draw matches
result = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    good,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Save output
os.makedirs("results", exist_ok=True)
cv2.imwrite("results/sift_matches.png", result)

print("=" * 40)
print("SIFT RESULTS")
print("=" * 40)
print("Image 1 keypoints :", len(kp1))
print("Image 2 keypoints :", len(kp2))
print("Good matches      :", len(good))
print("Execution time    : %.4f sec" % (end - start))
print("Output saved to   : results/sift_matches.png")
