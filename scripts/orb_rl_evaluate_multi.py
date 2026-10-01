import cv2
import os
import random
import numpy as np
import torch
import torch.nn as nn


# ==================================================
# SETTINGS
# ==================================================

DATASET = "hpatches"

MODEL = "scripts/orb_rl_policy_multi.pth"


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
# LOAD SEQUENCES
# ==================================================

sequences = []

for name in sorted(os.listdir(DATASET)):

    path = os.path.join(
        DATASET,
        name
    )

    if os.path.isdir(path):
        sequences.append(name)


# Same split used during training

random.seed(42)
random.shuffle(sequences)

split = int(len(sequences) * 0.8)

test_sequences = sequences[split:]


print("=" * 60)
print("MULTI-PAIR ORB vs ORB + RL EVALUATION")
print("=" * 60)

print("Test sequences:", len(test_sequences))

print()
print("Test sequences:")
print(test_sequences)


# ==================================================
# LOAD MODEL
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
# ORB
# ==================================================

orb = cv2.ORB_create()


# ==================================================
# KEYPOINT FEATURES
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

    if (
        descriptors is not None
        and other_descriptors is not None
    ):

        matches = bf.knnMatch(
            descriptors,
            other_descriptors,
            k=2
        )

    else:

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
# RL SELECTION
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
# MATCH + RANSAC
# ==================================================

def calculate_matches(
    kp1,
    des1,
    kp2,
    des2
):

    if des1 is None or des2 is None:
        return 0, 0


    if len(des1) < 2 or len(des2) < 2:
        return 0, 0


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

        return 0, 0


    good_matches = []


    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:

            good_matches.append(m)


    if len(good_matches) < 4:

        return len(good_matches), 0


    src_pts = np.float32([
        kp1[m.queryIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)


    dst_pts = np.float32([
        kp2[m.trainIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)


    try:

        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            5.0
        )

    except cv2.error:

        return len(good_matches), 0


    if mask is None:

        return len(good_matches), 0


    inliers = int(
        np.sum(mask)
    )


    return (
        len(good_matches),
        inliers
    )


# ==================================================
# TOTALS
# ==================================================

orb_keypoints_total = 0
orb_good_total = 0
orb_inliers_total = 0

rl_keypoints_total = 0
rl_good_total = 0
rl_inliers_total = 0

number_of_pairs = 0


# ==================================================
# EVALUATION
# ==================================================

print()
print("=" * 60)
print("STARTING EVALUATION")
print("=" * 60)


for sequence in test_sequences:

    for image_number in range(2, 7):

        path1 = os.path.join(
            DATASET,
            sequence,
            "1.ppm"
        )

        path2 = os.path.join(
            DATASET,
            sequence,
            str(image_number) + ".ppm"
        )


        img1 = cv2.imread(
            path1,
            cv2.IMREAD_GRAYSCALE
        )

        img2 = cv2.imread(
            path2,
            cv2.IMREAD_GRAYSCALE
        )


        if img1 is None or img2 is None:
            continue


        # ------------------------------------------
        # ORB
        # ------------------------------------------

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


        orb_good, orb_inliers = calculate_matches(
            kp1,
            des1,
            kp2,
            des2
        )


        # ------------------------------------------
        # RL FEATURES
        # ------------------------------------------

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


        # ------------------------------------------
        # RL SELECTION
        # ------------------------------------------

        (
            selected_kp1,
            selected_des1
        ) = select_keypoints(
            kp1,
            des1,
            states1
        )


        (
            selected_kp2,
            selected_des2
        ) = select_keypoints(
            kp2,
            des2,
            states2
        )


        # ------------------------------------------
        # RL MATCHING
        # ------------------------------------------

        rl_good, rl_inliers = calculate_matches(
            selected_kp1,
            selected_des1,
            selected_kp2,
            selected_des2
        )


        # ------------------------------------------
        # TOTALS
        # ------------------------------------------

        orb_keypoints_total += (
            len(kp1) + len(kp2)
        )

        orb_good_total += orb_good

        orb_inliers_total += orb_inliers


        rl_keypoints_total += (
            len(selected_kp1)
            +
            len(selected_kp2)
        )

        rl_good_total += rl_good

        rl_inliers_total += rl_inliers


        number_of_pairs += 1


        # ------------------------------------------
        # PRINT PAIR
        # ------------------------------------------

        print(
            sequence,
            "| 1->",
            image_number,
            "| ORB:",
            orb_good,
            "good /",
            orb_inliers,
            "inliers",
            "| RL:",
            rl_good,
            "good /",
            rl_inliers,
            "inliers",
            "| RL points:",
            len(selected_kp1)
            +
            len(selected_kp2)
        )


# ==================================================
# AVERAGES
# ==================================================

print()
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)


if number_of_pairs > 0:

    avg_orb_keypoints = (
        orb_keypoints_total /
        number_of_pairs
    )

    avg_orb_good = (
        orb_good_total /
        number_of_pairs
    )

    avg_orb_inliers = (
        orb_inliers_total /
        number_of_pairs
    )


    avg_rl_keypoints = (
        rl_keypoints_total /
        number_of_pairs
    )

    avg_rl_good = (
        rl_good_total /
        number_of_pairs
    )

    avg_rl_inliers = (
        rl_inliers_total /
        number_of_pairs
    )


    print()
    print("Number of test pairs:", number_of_pairs)

    print()
    print("ORB")
    print("----------------------")
    print(
        "Average keypoints :",
        round(avg_orb_keypoints, 2)
    )

    print(
        "Average good      :",
        round(avg_orb_good, 2)
    )

    print(
        "Average inliers   :",
        round(avg_orb_inliers, 2)
    )


    print()
    print("ORB + RL")
    print("----------------------")
    print(
        "Average keypoints :",
        round(avg_rl_keypoints, 2)
    )

    print(
        "Average good      :",
        round(avg_rl_good, 2)
    )

    print(
        "Average inliers   :",
        round(avg_rl_inliers, 2)
    )


    # ----------------------------------------------
    # REDUCTION
    # ----------------------------------------------

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


    print()
    print("COMPARISON")
    print("----------------------")

    print(
        "Keypoint reduction :",
        round(keypoint_reduction, 2),
        "%"
    )

    print(
        "Inlier change      :",
        round(inlier_change, 2),
        "%"
    )


print()
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
