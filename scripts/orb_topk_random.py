import cv2
import os
import numpy as np


DATASET = "hpatches"

# Same approximate total keypoint budget as the ORB + RL result
TARGET_TOTAL = 636

# Reproducible random baseline
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

orb = cv2.ORB_create()

sequences = sorted([
    name
    for name in os.listdir(DATASET)
    if os.path.isdir(os.path.join(DATASET, name))
])


def choose_budget(n1, n2, target):
    """
    Split the total target budget between image 1 and image 2
    according to the number of detected keypoints in each image.
    """
    total = n1 + n2

    if total == 0:
        return 0, 0

    k1 = int(round(target * n1 / total))
    k2 = target - k1

    k1 = min(k1, n1)
    k2 = min(k2, n2)

    return k1, k2


def select_topk(keypoints, descriptors, k):
    """
    Select top-k keypoints according to detector response.
    """
    if len(keypoints) <= k:
        return keypoints, descriptors

    indices = sorted(
        range(len(keypoints)),
        key=lambda i: keypoints[i].response,
        reverse=True
    )[:k]

    selected_kp = [keypoints[i] for i in indices]
    selected_des = descriptors[indices]

    return selected_kp, selected_des


def select_random(keypoints, descriptors, k):
    """
    Randomly select k keypoints with a fixed seed.
    """
    if len(keypoints) <= k:
        return keypoints, descriptors

    indices = rng.choice(
        len(keypoints),
        size=k,
        replace=False
    )

    indices = np.sort(indices)

    selected_kp = [keypoints[i] for i in indices]
    selected_des = descriptors[indices]

    return selected_kp, selected_des


def get_good_matches(des1, des2):

    if des1 is None or des2 is None:
        return []

    if len(des1) < 2 or len(des2) < 2:
        return []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    try:
        matches = bf.knnMatch(
            des1,
            des2,
            k=2
        )
    except cv2.error:
        return []

    good = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good.append(m)

    return good


def get_ransac_inliers(kp1, kp2, matches):

    if len(matches) < 4:
        return 0

    src_pts = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    dst_pts = np.float32([
        kp2[m.trainIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    try:

        _, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            5.0
        )

    except cv2.error:
        return 0

    if mask is None:
        return 0

    return int(np.sum(mask))


def calculate_mma(kp1, kp2, matches, H):

    if len(matches) == 0:
        return 0.0, 0.0, 0.0

    points = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    projected = cv2.perspectiveTransform(
        points,
        H
    )

    count_1 = 0
    count_3 = 0
    count_5 = 0

    for i, m in enumerate(matches):

        px = projected[i][0][0]
        py = projected[i][0][1]

        ax = kp2[m.trainIdx].pt[0]
        ay = kp2[m.trainIdx].pt[1]

        error = np.sqrt(
            (px - ax) ** 2 +
            (py - ay) ** 2
        )

        if error <= 1:
            count_1 += 1

        if error <= 3:
            count_3 += 1

        if error <= 5:
            count_5 += 1

    total = len(matches)

    return (
        count_1 / total,
        count_3 / total,
        count_5 / total
    )


def evaluate(method_name, selector):

    all_mma1 = []
    all_mma3 = []
    all_mma5 = []

    all_inliers = []
    all_good = []
    all_keypoints = []

    total_pairs = 0

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

            kp1, des1 = orb.detectAndCompute(
                img1,
                None
            )

            kp2, des2 = orb.detectAndCompute(
                img2,
                None
            )

            if des1 is None or des2 is None:
                continue

            # Same total budget as RL, distributed between both images
            k1, k2 = choose_budget(
                len(kp1),
                len(kp2),
                TARGET_TOTAL
            )

            kp1_sel, des1_sel = selector(
                kp1,
                des1,
                k1
            )

            kp2_sel, des2_sel = selector(
                kp2,
                des2,
                k2
            )

            selected_total = (
                len(kp1_sel) +
                len(kp2_sel)
            )

            all_keypoints.append(
                selected_total
            )

            good_matches = get_good_matches(
                des1_sel,
                des2_sel
            )

            all_good.append(
                len(good_matches)
            )

            inliers = get_ransac_inliers(
                kp1_sel,
                kp2_sel,
                good_matches
            )

            all_inliers.append(
                inliers
            )

            mma1, mma3, mma5 = calculate_mma(
                kp1_sel,
                kp2_sel,
                good_matches,
                H
            )

            all_mma1.append(
                mma1 * 100
            )

            all_mma3.append(
                mma3 * 100
            )

            all_mma5.append(
                mma5 * 100
            )

            total_pairs += 1

    print()
    print("=" * 65)
    print(method_name)
    print("=" * 65)

    print(
        "Total evaluated pairs:",
        total_pairs
    )

    print(
        "Average selected keypoints:",
        round(np.mean(all_keypoints), 2)
    )

    print(
        "Average good matches:",
        round(np.mean(all_good), 2)
    )

    print(
        "Average RANSAC inliers:",
        round(np.mean(all_inliers), 2)
    )

    print(
        "MMA @ 1 px:",
        round(np.mean(all_mma1), 2),
        "%"
    )

    print(
        "MMA @ 3 px:",
        round(np.mean(all_mma3), 2),
        "%"
    )

    print(
        "MMA @ 5 px:",
        round(np.mean(all_mma5), 2),
        "%"
    )


print("=" * 65)
print("ORB TOP-K AND RANDOM BASELINES")
print("=" * 65)
print("Total sequences:", len(sequences))
print("Expected pairs :", len(sequences) * 5)
print("Target keypoints per pair:", TARGET_TOTAL)
print()


# ============================================================
# TOP-K BASELINE
# ============================================================

evaluate(
    "ORB TOP-K BY RESPONSE",
    select_topk
)


# ============================================================
# RANDOM BASELINE
# ============================================================

evaluate(
    "ORB RANDOM SUBSAMPLE",
    select_random
)


print()
print("=" * 65)
print("ORB BASELINE EXPERIMENTS COMPLETE")
print("=" * 65)
