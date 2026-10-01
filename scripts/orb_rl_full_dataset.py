import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ==================================================
# SETTINGS
# ==================================================

DATASET = "hpatches"
MODEL = "scripts/orb_rl_policy_10000_full.pth"


# ==================================================
# RL POLICY
# ==================================================

class KeypointPolicy(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 2)
        )

    def forward(self, state):
        return self.network(state)


# ==================================================
# LOAD RL MODEL
# ==================================================

policy = KeypointPolicy()

policy.load_state_dict(
    torch.load(
        MODEL,
        weights_only=True
    )
)

policy.eval()


# ==================================================
# ALL HPATCHES SEQUENCES
# ==================================================

sequences = sorted([
    name
    for name in os.listdir(DATASET)
    if os.path.isdir(
        os.path.join(DATASET, name)
    )
])


print("=" * 60)
print("FULL HPATCHES: ORB vs ORB + RL")
print("=" * 60)

print("Total sequences:", len(sequences))
print("Total pairs:", len(sequences) * 5)


# ==================================================
# ORB
# ==================================================

orb = cv2.ORB_create()


# ==================================================
# FEATURE STATE
# ==================================================

def get_features(
    keypoints,
    descriptors,
    other_descriptors,
    width,
    height
):

    features = []

    bf = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )

    try:

        matches = bf.knnMatch(
            descriptors,
            other_descriptors,
            k=2
        )

    except cv2.error:

        matches = []


    for i, kp in enumerate(keypoints):

        best_distance = 256.0
        second_distance = 256.0
        ratio = 1.0

        if i < len(matches):

            pair = matches[i]

            if len(pair) >= 2:

                m, n = pair

                best_distance = m.distance
                second_distance = n.distance

                if n.distance > 0:

                    ratio = (
                        m.distance /
                        n.distance
                    )


        state = [

            kp.pt[0] / width,
            kp.pt[1] / height,

            kp.size / 100.0,
            kp.angle / 360.0,
            kp.response,

            best_distance / 256.0,
            second_distance / 256.0,
            ratio
        ]

        features.append(state)


    return torch.tensor(
        features,
        dtype=torch.float32
    )


# ==================================================
# RL KEYPOINT SELECTION
# ==================================================

def select_keypoints(
    keypoints,
    descriptors,
    states
):

    selected_keypoints = []
    selected_descriptors = []


    for i in range(len(keypoints)):

        state = states[i].unsqueeze(0)

        with torch.no_grad():

            output = policy(state)

            probability = torch.softmax(
                output,
                dim=1
            )

            action = torch.argmax(
                probability,
                dim=1
            ).item()


        # 1 = KEEP

        if action == 1:

            selected_keypoints.append(
                keypoints[i]
            )

            selected_descriptors.append(
                descriptors[i]
            )


    if len(selected_descriptors) > 0:

        selected_descriptors = np.array(
            selected_descriptors,
            dtype=np.uint8
        )

    else:

        selected_descriptors = None


    return (
        selected_keypoints,
        selected_descriptors
    )


# ==================================================
# MATCHING
# ==================================================

def get_good_matches(
    des1,
    des2
):

    if des1 is None or des2 is None:
        return []

    if len(des1) < 2 or len(des2) < 2:
        return []

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

        return []


    good_matches = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:

            good_matches.append(m)


    return good_matches


# ==================================================
# RANSAC
# ==================================================

def get_ransac_inliers(
    kp1,
    kp2,
    matches
):

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


# ==================================================
# MMA
# ==================================================

def calculate_mma(
    kp1,
    kp2,
    matches,
    H
):

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
            (px - ax) ** 2
            +
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


# ==================================================
# TOTALS
# ==================================================

pairs = 0

orb_keypoints = 0
rl_keypoints = 0

orb_good = 0
rl_good = 0

orb_inliers = 0
rl_inliers = 0

orb_mma1 = []
orb_mma3 = []
orb_mma5 = []

rl_mma1 = []
rl_mma3 = []
rl_mma5 = []


# ==================================================
# MAIN LOOP
# ==================================================

print()
print("Starting full-dataset evaluation...")

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

        H_path = os.path.join(
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
                H_path
            ).reshape(3, 3)

        except Exception:

            continue


        # ==================================================
        # ORB
        # ==================================================

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


        orb_keypoints += (
            len(kp1) +
            len(kp2)
        )


        orb_matches = get_good_matches(
            des1,
            des2
        )


        orb_good += len(
            orb_matches
        )


        orb_inliers += get_ransac_inliers(
            kp1,
            kp2,
            orb_matches
        )


        (
            m1,
            m3,
            m5
        ) = calculate_mma(
            kp1,
            kp2,
            orb_matches,
            H
        )


        orb_mma1.append(m1)
        orb_mma3.append(m3)
        orb_mma5.append(m5)


        # ==================================================
        # RL FEATURES
        # ==================================================

        states1 = get_features(
            kp1,
            des1,
            des2,
            img1.shape[1],
            img1.shape[0]
        )


        states2 = get_features(
            kp2,
            des2,
            des1,
            img2.shape[1],
            img2.shape[0]
        )


        # ==================================================
        # RL SELECTION
        # ==================================================

        (
            rl_kp1,
            rl_des1
        ) = select_keypoints(
            kp1,
            des1,
            states1
        )


        (
            rl_kp2,
            rl_des2
        ) = select_keypoints(
            kp2,
            des2,
            states2
        )


        rl_keypoints += (
            len(rl_kp1) +
            len(rl_kp2)
        )


        # ==================================================
        # RL MATCHING
        # ==================================================

        rl_matches = get_good_matches(
            rl_des1,
            rl_des2
        )


        rl_good += len(
            rl_matches
        )


        rl_inliers += get_ransac_inliers(
            rl_kp1,
            rl_kp2,
            rl_matches
        )


        (
            r1,
            r3,
            r5
        ) = calculate_mma(
            rl_kp1,
            rl_kp2,
            rl_matches,
            H
        )


        rl_mma1.append(r1)
        rl_mma3.append(r3)
        rl_mma5.append(r5)


        pairs += 1


        if pairs % 50 == 0:

            print(
                "Processed pairs:",
                pairs
            )


# ==================================================
# FINAL RESULTS
# ==================================================

avg_orb_keypoints = (
    orb_keypoints / pairs
)

avg_rl_keypoints = (
    rl_keypoints / pairs
)

avg_orb_good = (
    orb_good / pairs
)

avg_rl_good = (
    rl_good / pairs
)

avg_orb_inliers = (
    orb_inliers / pairs
)

avg_rl_inliers = (
    rl_inliers / pairs
)


orb_mma1_final = np.mean(
    orb_mma1
) * 100

orb_mma3_final = np.mean(
    orb_mma3
) * 100

orb_mma5_final = np.mean(
    orb_mma5
) * 100


rl_mma1_final = np.mean(
    rl_mma1
) * 100

rl_mma3_final = np.mean(
    rl_mma3
) * 100

rl_mma5_final = np.mean(
    rl_mma5
) * 100


keypoint_reduction = (
    1 -
    (
        avg_rl_keypoints /
        avg_orb_keypoints
    )
) * 100


inlier_change = (
    (
        avg_rl_inliers /
        max(avg_orb_inliers, 1)
    )
    - 1
) * 100


# ==================================================
# PRINT
# ==================================================

print()
print("=" * 60)
print("FULL HPATCHES: ORB vs ORB + RL")
print("=" * 60)

print(
    "Total evaluated pairs:",
    pairs
)

print()
print("AVERAGE KEYPOINTS")
print(
    "ORB     :",
    round(avg_orb_keypoints, 2)
)

print(
    "ORB + RL:",
    round(avg_rl_keypoints, 2)
)

print()
print("AVERAGE GOOD MATCHES")
print(
    "ORB     :",
    round(avg_orb_good, 2)
)

print(
    "ORB + RL:",
    round(avg_rl_good, 2)
)

print()
print("AVERAGE RANSAC INLIERS")
print(
    "ORB     :",
    round(avg_orb_inliers, 2)
)

print(
    "ORB + RL:",
    round(avg_rl_inliers, 2)
)

print()
print("MMA @ 1 px")
print(
    "ORB     :",
    round(orb_mma1_final, 2),
    "%"
)

print(
    "ORB + RL:",
    round(rl_mma1_final, 2),
    "%"
)

print()
print("MMA @ 3 px")
print(
    "ORB     :",
    round(orb_mma3_final, 2),
    "%"
)

print(
    "ORB + RL:",
    round(rl_mma3_final, 2),
    "%"
)

print()
print("MMA @ 5 px")
print(
    "ORB     :",
    round(orb_mma5_final, 2),
    "%"
)

print(
    "ORB + RL:",
    round(rl_mma5_final, 2),
    "%"
)

print()
print(
    "Keypoint reduction:",
    round(keypoint_reduction, 2),
    "%"
)

print(
    "Inlier change:",
    round(inlier_change, 2),
    "%"
)


# ==================================================
# GRAPH 1 — MMA BEFORE VS AFTER
# ==================================================

labels = [
    "1 px",
    "3 px",
    "5 px"
]

orb_values = [
    orb_mma1_final,
    orb_mma3_final,
    orb_mma5_final
]

rl_values = [
    rl_mma1_final,
    rl_mma3_final,
    rl_mma5_final
]

x = np.arange(
    len(labels)
)

width = 0.35


plt.figure(
    figsize=(9, 5)
)

plt.bar(
    x - width / 2,
    orb_values,
    width,
    label="ORB"
)

plt.bar(
    x + width / 2,
    rl_values,
    width,
    label="ORB + RL"
)

plt.xticks(
    x,
    labels
)

plt.ylabel(
    "Mean Matching Accuracy (%)"
)

plt.xlabel(
    "Pixel error threshold"
)

plt.title(
    "ORB vs ORB + RL on Complete HPatches Dataset"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/full_dataset_orb_vs_rl_mma.png",
    dpi=300
)

plt.close()


# ==================================================
# GRAPH 2 — KEYPOINTS BEFORE VS AFTER
# ==================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    ["ORB", "ORB + RL"],
    [
        avg_orb_keypoints,
        avg_rl_keypoints
    ]
)

plt.ylabel(
    "Average keypoints per image pair"
)

plt.title(
    "Keypoint Reduction on Complete HPatches Dataset"
)

plt.tight_layout()

plt.savefig(
    "results/full_dataset_orb_vs_rl_keypoints.png",
    dpi=300
)

plt.close()


print()
print("Graphs saved:")
print(
    "results/full_dataset_orb_vs_rl_mma.png"
)

print(
    "results/full_dataset_orb_vs_rl_keypoints.png"
)

print()
print("=" * 60)
print("FULL DATASET EVALUATION COMPLETE")
print("=" * 60)
