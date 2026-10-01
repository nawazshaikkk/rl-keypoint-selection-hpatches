import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

DATASET = "hpatches"

brisk = cv2.BRISK_create()

# Get all HPatches sequences
sequences = sorted([
    name
    for name in os.listdir(DATASET)
    if os.path.isdir(os.path.join(DATASET, name))
])

print("=" * 60)
print("BRISK FULL HPATCHES EVALUATION")
print("=" * 60)

print("Total sequences:", len(sequences))
print("Expected pairs :", len(sequences) * 5)

total_pairs = 0
total_keypoints = 0
total_good_matches = 0
total_inliers = 0

mma_1 = []
mma_3 = []
mma_5 = []

for sequence in sequences:

    folder = os.path.join(DATASET, sequence)

    for image_number in range(2, 7):

        image1_path = os.path.join(
            folder,
            "1.ppm"
        )

        image2_path = os.path.join(
            folder,
            str(image_number) + ".ppm"
        )

        homography_path = os.path.join(
            folder,
            "H_1_" + str(image_number)
        )

        # Load images
        img1 = cv2.imread(
            image1_path,
            cv2.IMREAD_GRAYSCALE
        )

        img2 = cv2.imread(
            image2_path,
            cv2.IMREAD_GRAYSCALE
        )

        if img1 is None or img2 is None:
            continue

        # Load ground-truth homography
        try:
            H = np.loadtxt(
                homography_path
            ).reshape(3, 3)
        except Exception:
            continue

        # BRISK
        kp1, des1 = brisk.detectAndCompute(
            img1,
            None
        )

        kp2, des2 = brisk.detectAndCompute(
            img2,
            None
        )

        if des1 is None or des2 is None:
            continue

        total_keypoints += (
            len(kp1) + len(kp2)
        )

        # BF matcher
        bf = cv2.BFMatcher(
            cv2.NORM_HAMMING
        )

        try:
            matches = bf.knnMatch(
                des1,
                des2,
                k=2
            )
        except cv2.error:
            continue

        # Lowe ratio test
        good_matches = []

        for pair in matches:

            if len(pair) < 2:
                continue

            m, n = pair

            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        total_good_matches += len(
            good_matches
        )

        # RANSAC
        inliers = 0

        if len(good_matches) >= 4:

            src_pts = np.float32([
                kp1[m.queryIdx].pt
                for m in good_matches
            ]).reshape(-1, 1, 2)

            dst_pts = np.float32([
                kp2[m.trainIdx].pt
                for m in good_matches
            ]).reshape(-1, 1, 2)

            try:

                _, mask = cv2.findHomography(
                    src_pts,
                    dst_pts,
                    cv2.RANSAC,
                    5.0
                )

                if mask is not None:
                    inliers = int(
                        np.sum(mask)
                    )

            except cv2.error:
                inliers = 0

        total_inliers += inliers

        # MMA using ground-truth homography
        if len(good_matches) > 0:

            points1 = np.float32([
                kp1[m.queryIdx].pt
                for m in good_matches
            ]).reshape(-1, 1, 2)

            projected = cv2.perspectiveTransform(
                points1,
                H
            )

            count_1 = 0
            count_3 = 0
            count_5 = 0

            for i, m in enumerate(good_matches):

                predicted_x = projected[i][0][0]
                predicted_y = projected[i][0][1]

                actual_x = kp2[m.trainIdx].pt[0]
                actual_y = kp2[m.trainIdx].pt[1]

                error = np.sqrt(
                    (predicted_x - actual_x) ** 2
                    +
                    (predicted_y - actual_y) ** 2
                )

                if error <= 1:
                    count_1 += 1

                if error <= 3:
                    count_3 += 1

                if error <= 5:
                    count_5 += 1

            total = len(good_matches)

            mma_1.append(
                count_1 / total
            )

            mma_3.append(
                count_3 / total
            )

            mma_5.append(
                count_5 / total
            )

        total_pairs += 1

        if total_pairs % 50 == 0:
            print(
                "Processed pairs:",
                total_pairs
            )

# Final results
print()
print("=" * 60)
print("BRISK FULL DATASET RESULTS")
print("=" * 60)

print(
    "Total evaluated pairs:",
    total_pairs
)

if total_pairs > 0:

    avg_keypoints = (
        total_keypoints / total_pairs
    )

    avg_good = (
        total_good_matches / total_pairs
    )

    avg_inliers = (
        total_inliers / total_pairs
    )

    avg_mma_1 = (
        np.mean(mma_1) * 100
    )

    avg_mma_3 = (
        np.mean(mma_3) * 100
    )

    avg_mma_5 = (
        np.mean(mma_5) * 100
    )

    print(
        "Average keypoints:",
        round(avg_keypoints, 2)
    )

    print(
        "Average good matches:",
        round(avg_good, 2)
    )

    print(
        "Average RANSAC inliers:",
        round(avg_inliers, 2)
    )

    print()
    print(
        "MMA @ 1 px:",
        round(avg_mma_1, 2),
        "%"
    )

    print(
        "MMA @ 3 px:",
        round(avg_mma_3, 2),
        "%"
    )

    print(
        "MMA @ 5 px:",
        round(avg_mma_5, 2),
        "%"
    )

    # Accuracy graph
    plt.figure(figsize=(9, 5))

    plt.bar(
        ["1 px", "3 px", "5 px"],
        [
            avg_mma_1,
            avg_mma_3,
            avg_mma_5
        ]
    )

    plt.xlabel(
        "Pixel error threshold"
    )

    plt.ylabel(
        "Mean Matching Accuracy (%)"
    )

    plt.title(
        "BRISK Performance on Complete HPatches Dataset"
    )

    plt.tight_layout()

    plt.savefig(
        "results/brisk_full_dataset_mma.png",
        dpi=300
    )

    plt.close()

    print()
    print(
        "Graph saved:",
        "results/brisk_full_dataset_mma.png"
    )

print()
print("=" * 60)
print("BRISK FULL DATASET EVALUATION COMPLETE")
print("=" * 60)
