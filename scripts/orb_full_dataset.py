import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


DATASET = "hpatches"

orb = cv2.ORB_create()

sequences = sorted([
    name
    for name in os.listdir(DATASET)
    if os.path.isdir(os.path.join(DATASET, name))
])

print("=" * 60)
print("ORB FULL HPATCHES EVALUATION")
print("=" * 60)

print("Total sequences:", len(sequences))
print("Expected pairs :", len(sequences) * 5)


sequence_names = []

sequence_mma1 = []
sequence_mma3 = []
sequence_mma5 = []

sequence_inliers = []
sequence_good = []
sequence_keypoints = []

total_pairs = 0


for sequence in sequences:

    folder = os.path.join(DATASET, sequence)

    seq_mma1 = []
    seq_mma3 = []
    seq_mma5 = []

    seq_inliers = []
    seq_good = []
    seq_keypoints = []

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

        try:
            H = np.loadtxt(homography_path).reshape(3, 3)
        except Exception:
            continue

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            continue

        seq_keypoints.append(
            len(kp1) + len(kp2)
        )

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        try:
            matches = bf.knnMatch(
                des1,
                des2,
                k=2
            )
        except cv2.error:
            continue

        good_matches = []

        for pair in matches:

            if len(pair) < 2:
                continue

            m, n = pair

            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        seq_good.append(len(good_matches))

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
                    inliers = int(np.sum(mask))

            except cv2.error:
                inliers = 0

        seq_inliers.append(inliers)

        if len(good_matches) > 0:

            points = np.float32([
                kp1[m.queryIdx].pt
                for m in good_matches
            ]).reshape(-1, 1, 2)

            projected = cv2.perspectiveTransform(
                points,
                H
            )

            c1 = 0
            c3 = 0
            c5 = 0

            for i, m in enumerate(good_matches):

                px = projected[i][0][0]
                py = projected[i][0][1]

                ax = kp2[m.trainIdx].pt[0]
                ay = kp2[m.trainIdx].pt[1]

                error = np.sqrt(
                    (px - ax) ** 2 +
                    (py - ay) ** 2
                )

                if error <= 1:
                    c1 += 1

                if error <= 3:
                    c3 += 1

                if error <= 5:
                    c5 += 1

            total = len(good_matches)

            seq_mma1.append(c1 / total)
            seq_mma3.append(c3 / total)
            seq_mma5.append(c5 / total)

        total_pairs += 1

    if len(seq_mma1) > 0:

        sequence_names.append(sequence)

        sequence_mma1.append(
            np.mean(seq_mma1) * 100
        )

        sequence_mma3.append(
            np.mean(seq_mma3) * 100
        )

        sequence_mma5.append(
            np.mean(seq_mma5) * 100
        )

        sequence_inliers.append(
            np.mean(seq_inliers)
        )

        sequence_good.append(
            np.mean(seq_good)
        )

        sequence_keypoints.append(
            np.mean(seq_keypoints)
        )


print()
print("=" * 60)
print("ORB FULL DATASET RESULTS")
print("=" * 60)

print("Total evaluated pairs:", total_pairs)

print(
    "Average keypoints:",
    round(np.mean(sequence_keypoints), 2)
)

print(
    "Average good matches:",
    round(np.mean(sequence_good), 2)
)

print(
    "Average RANSAC inliers:",
    round(np.mean(sequence_inliers), 2)
)

print(
    "MMA @ 1 px:",
    round(np.mean(sequence_mma1), 2),
    "%"
)

print(
    "MMA @ 3 px:",
    round(np.mean(sequence_mma3), 2),
    "%"
)

print(
    "MMA @ 5 px:",
    round(np.mean(sequence_mma5), 2),
    "%"
)


# ==================================================
# GRAPH 1 — FULL DATASET MMA
# ==================================================

plt.figure(figsize=(8, 5))

plt.bar(
    ["1 px", "3 px", "5 px"],
    [
        np.mean(sequence_mma1),
        np.mean(sequence_mma3),
        np.mean(sequence_mma5)
    ]
)

plt.xlabel("Pixel error threshold")
plt.ylabel("Mean Matching Accuracy (%)")
plt.title("ORB Performance on Complete HPatches Dataset")

plt.tight_layout()

plt.savefig(
    "results/orb_full_dataset_mma.png",
    dpi=300
)

plt.close()


# ==================================================
# GRAPH 2 — ACCURACY ACROSS ALL 116 SEQUENCES
# ==================================================

plt.figure(figsize=(14, 5))

plt.plot(
    sequence_mma1,
    label="MMA @ 1 px"
)

plt.plot(
    sequence_mma3,
    label="MMA @ 3 px"
)

plt.plot(
    sequence_mma5,
    label="MMA @ 5 px"
)

plt.xlabel("HPatches sequence")
plt.ylabel("MMA (%)")
plt.title("ORB Accuracy Across HPatches Sequences")

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/orb_full_dataset_sequence_accuracy.png",
    dpi=300
)

plt.close()


# ==================================================
# GRAPH 3 — KEYPOINTS ACROSS ALL SEQUENCES
# ==================================================

plt.figure(figsize=(14, 5))

plt.plot(
    sequence_keypoints
)

plt.xlabel("HPatches sequence")
plt.ylabel("Average keypoints per pair")
plt.title("ORB Keypoints Across HPatches Sequences")

plt.tight_layout()

plt.savefig(
    "results/orb_full_dataset_keypoints.png",
    dpi=300
)

plt.close()


print()
print("Graphs saved:")
print("results/orb_full_dataset_mma.png")
print("results/orb_full_dataset_sequence_accuracy.png")
print("results/orb_full_dataset_keypoints.png")

print()
print("=" * 60)
print("COMPLETE HPATCHES ORB EVALUATION FINISHED")
print("=" * 60)
        
