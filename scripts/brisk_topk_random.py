import cv2
import os
import numpy as np


DATASET = "hpatches"

# Same approximate total keypoint budget as the
# 10,000-episode BRISK + RL result.
TARGET_TOTAL = 1388

# Fixed seed so the random experiment is reproducible.
rng = np.random.default_rng(42)

brisk = cv2.BRISK_create()

sequences = sorted([
    name
    for name in os.listdir(DATASET)
    if os.path.isdir(os.path.join(DATASET, name))
])


def choose_budget(n1, n2, target):
    """
    Split the total keypoint budget between image 1 and image 2
    according to the number of detected keypoints in each image.
    """
    total = n1 + n2

    if total == 0:
        return 0, 0

    k1 = int(round(target * n1 / total))
    k2 = target - k1

    k1 = min(k1, n1)
    k2 = min(k2, n2)

    # If clipping one side reduced the total budget,
    # try to assign the remainder to the other side.
    current = k1 + k2

    if current < target:
        remaining = target - current

        extra1 = min(remaining, n1 - k1)
        k1 += extra1
        remaining -= extra1

        extra2 = min(remaining, n2 - k2)
        k2 += extra2

    return k1, k2


def select_topk(keypoints, descriptors, k):
    """
    Keep the k keypoints having the highest BRISK detector response.
    """
    if len(keypoints) <= k:
        return keypoints, descriptors

    indices = np.argsort(
        [-kp.response for kp in keypoints]
    )[:k]

    indices = np.asarray(indices, dtype=int)

    selected_kp = [
        keypoints[i]
        for i in indices
    ]

    selected_des = descriptors[indices]

    return selected_kp, selected_des


def select_random(keypoints, descriptors, k):
    """
    Randomly keep k keypoints using the fixed random generator.
    """
    if len(keypoints) <= k:
        return keypoints, descriptors

    indices = rng.choice(
        len(keypoints),
        size=k,
        replace=False
    )

    indices = np.sort(indices)

    selected_kp = [
        keypoints[i]
        for i in indices
    ]

    selected_des = descriptors[indices]

    return selected_kp, selected_des


def lowe_matches(des1, des2):

    if des1 is None or des2 is None:
        return []

    if len(des1) < 2 or len(des2) < 2:
        return []

    bf = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False
    )

    try:
        knn = bf.knnMatch(
            des1,
            des2,
            k=2
        )
    except cv2.error:
        return []

    good = []

    for pair in knn:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good.append(m)

    return good


def compute_ransac_inliers(kp1, kp2, matches):

    if len(matches) < 4:
        return 0

    pts1 = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    pts2 = np.float32([
        kp2[m.trainIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    try:

        _, mask = cv2.findHomography(
            pts1,
            pts2,
            cv2.RANSAC,
            5.0
        )

        if mask is None:
            return 0

        return int(mask.sum())

    except cv2.error:
        return 0


def compute_mma(kp1, kp2, matches, H):

    if H is None:
        return 0.0, 0.0, 0.0

    if len(matches) == 0:
        return 0.0, 0.0, 0.0

    errors = []

    for m in matches:

        p1 = np.array(
            [[kp1[m.queryIdx].pt]],
            dtype=np.float32
        )

        projected = cv2.perspectiveTransform(
            p1,
            H
        )[0][0]

        actual = np.array(
            kp2[m.trainIdx].pt,
            dtype=np.float32
        )

        error = np.linalg.norm(
            projected - actual
        )

        errors.append(error)

    errors = np.asarray(errors)

    mma1 = np.mean(errors <= 1.0) * 100.0
    mma3 = np.mean(errors <= 3.0) * 100.0
    mma5 = np.mean(errors <= 5.0) * 100.0

    return mma1, mma3, mma5


def evaluate(method_name, selector):

    total_pairs = 0

    total_keypoints = 0
    total_good_matches = 0
    total_inliers = 0

    mma_1 = []
    mma_3 = []
    mma_5 = []

    for sequence in sequences:

        folder = os.path.join(
            DATASET,
            sequence
        )

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
                H = np.loadtxt(
                    homography_path
                ).reshape(3, 3)
            except Exception:
                continue

            kp1_all, des1_all = brisk.detectAndCompute(
                img1,
                None
            )

            kp2_all, des2_all = brisk.detectAndCompute(
                img2,
                None
            )

            if des1_all is None or des2_all is None:
                continue

            # Same approximate combined keypoint budget
            # as the 10,000-episode BRISK + RL run.
            k1, k2 = choose_budget(
                len(kp1_all),
                len(kp2_all),
                TARGET_TOTAL
            )

            kp1, des1 = selector(
                kp1_all,
                des1_all,
                k1
            )

            kp2, des2 = selector(
                kp2_all,
                des2_all,
                k2
            )

            total_keypoints += (
                len(kp1) + len(kp2)
            )

            good_matches = lowe_matches(
                des1,
                des2
            )

            total_good_matches += len(
                good_matches
            )

            inliers = compute_ransac_inliers(
                kp1,
                kp2,
                good_matches
            )

            total_inliers += inliers

            m1, m3, m5 = compute_mma(
                kp1,
                kp2,
                good_matches,
                H
            )

            mma_1.append(m1 / 100.0)
            mma_3.append(m3 / 100.0)
            mma_5.append(m5 / 100.0)

            total_pairs += 1

            if total_pairs % 50 == 0:
                print(
                    method_name,
                    "- processed pairs:",
                    total_pairs
                )

    print()
    print("=" * 65)
    print(method_name)
    print("=" * 65)

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

        avg_mma1 = (
            np.mean(mma_1) * 100
        )

        avg_mma3 = (
            np.mean(mma_3) * 100
        )

        avg_mma5 = (
            np.mean(mma_5) * 100
        )

        print(
            "Average selected keypoints:",
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

        print(
            "MMA @ 1 px:",
            round(avg_mma1, 2),
            "%"
        )

        print(
            "MMA @ 3 px:",
            round(avg_mma3, 2),
            "%"
        )

        print(
            "MMA @ 5 px:",
            round(avg_mma5, 2),
            "%"
        )


print("=" * 65)
print("BRISK TOP-K AND RANDOM BASELINES")
print("=" * 65)
print("Total sequences:", len(sequences))
print("Expected pairs :", len(sequences) * 5)
print("Target keypoints per pair:", TARGET_TOTAL)


# ============================================================
# TOP-K
# ============================================================

evaluate(
    "BRISK TOP-K BY RESPONSE",
    select_topk
)


# ============================================================
# RANDOM
# ============================================================

evaluate(
    "BRISK RANDOM SUBSAMPLE",
    select_random
)


print()
print("=" * 65)
print("BRISK BASELINE EXPERIMENTS COMPLETE")
print("=" * 65)
