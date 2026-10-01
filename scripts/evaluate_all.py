import cv2
import os
import time
import csv
import numpy as np

DATASET = "hpatches"
OUTPUT = "results/all_results.csv"

# ---------------------------------------------------------
# Descriptor creation
# ---------------------------------------------------------

def get_features(method, image):
    if method == "SIFT":
        detector = cv2.SIFT_create()
        return detector.detectAndCompute(image, None)

    elif method == "ORB":
        detector = cv2.ORB_create()
        return detector.detectAndCompute(image, None)

    elif method == "BRISK":
        detector = cv2.BRISK_create()
        return detector.detectAndCompute(image, None)

    elif method == "LATCH":
        # LATCH needs a separate keypoint detector.
        # We use SIFT keypoints to match your report.
        detector = cv2.SIFT_create()
        keypoints = detector.detect(image, None)

        latch = cv2.xfeatures2d.LATCH_create()
        return latch.compute(image, keypoints)


# ---------------------------------------------------------
# Matching
# ---------------------------------------------------------

def match_descriptors(method, des1, des2):

    if des1 is None or des2 is None:
        return []

    if method == "SIFT":
        bf = cv2.BFMatcher(cv2.NORM_L2)
        knn_matches = bf.knnMatch(des1, des2, k=2)

        good = []

        for pair in knn_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        return good

    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn_matches = bf.knnMatch(des1, des2, k=2)

        good = []

        for pair in knn_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        return good


# ---------------------------------------------------------
# Evaluate matches using HPatches homography
# ---------------------------------------------------------

def evaluate_matches(kp1, kp2, matches, H):

    if len(matches) == 0:
        return 0, 0, 0, 0

    pts1 = np.float32(
        [kp1[m.queryIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    pts2 = np.float32(
        [kp2[m.trainIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    # Transform points from image 1 to image 2
    projected = cv2.perspectiveTransform(pts1, H)

    errors = np.linalg.norm(
        projected.reshape(-1, 2) -
        pts2.reshape(-1, 2),
        axis=1
    )

    # Correct matches
    correct_1px = np.sum(errors <= 1.0)
    correct_3px = np.sum(errors <= 3.0)

    precision = correct_3px / len(matches)

    mma_1px = correct_1px / len(matches)
    mma_3px = correct_3px / len(matches)

    return precision, mma_1px, mma_3px, np.mean(errors)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

methods = ["SIFT", "ORB", "BRISK", "LATCH"]

os.makedirs("results", exist_ok=True)

rows = []

folders = sorted([
    f for f in os.listdir(DATASET)
    if os.path.isdir(os.path.join(DATASET, f))
])

print("=" * 70)
print("HPATCHES FULL DATASET EVALUATION")
print("=" * 70)

print("Number of sequences:", len(folders))
print()

for folder_number, folder in enumerate(folders, start=1):

    folder_path = os.path.join(DATASET, folder)

    print(
        f"[{folder_number}/{len(folders)}] Processing {folder}"
    )

    img1_path = os.path.join(folder_path, "1.ppm")

    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None:
        print("  ERROR: Could not read 1.ppm")
        continue

    # Compare image 1 with images 2-6
    for image_number in range(2, 7):

        img2_path = os.path.join(
            folder_path,
            f"{image_number}.ppm"
        )

        H_path = os.path.join(
            folder_path,
            f"H_1_{image_number}"
        )

        img2 = cv2.imread(
            img2_path,
            cv2.IMREAD_GRAYSCALE
        )

        if img2 is None:
            print(
                f"  Skipping image {image_number}: "
                "image not found"
            )
            continue

        if not os.path.exists(H_path):
            print(
                f"  Skipping image {image_number}: "
                "homography not found"
            )
            continue

        # Read homography
        H = np.loadtxt(H_path)

        for method in methods:

            try:

                start = time.time()

                kp1, des1 = get_features(
                    method,
                    img1
                )

                kp2, des2 = get_features(
                    method,
                    img2
                )

                matches = match_descriptors(
                    method,
                    des1,
                    des2
                )

                end = time.time()

                execution_time = end - start

                precision, mma1, mma3, mean_error = \
                    evaluate_matches(
                        kp1,
                        kp2,
                        matches,
                        H
                    )

                rows.append([
                    folder,
                    f"1->{image_number}",
                    method,
                    len(kp1),
                    len(kp2),
                    len(matches),
                    precision,
                    mma1,
                    mma3,
                    mean_error,
                    execution_time
                ])

            except Exception as e:

                print(
                    f"  ERROR: {method} "
                    f"on {folder} 1->{image_number}: {e}"
                )


# ---------------------------------------------------------
# Save CSV
# ---------------------------------------------------------

with open(
    OUTPUT,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Sequence",
        "Pair",
        "Method",
        "Keypoints1",
        "Keypoints2",
        "GoodMatches",
        "Precision",
        "MMA@1px",
        "MMA@3px",
        "MeanError",
        "TimeSeconds"
    ])

    writer.writerows(rows)


print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print("Total evaluations:", len(rows))
print("Results saved to:", OUTPUT)
