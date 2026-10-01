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

    path = os.path.join(DATASET, name)

    if os.path.isdir(path):
        sequences.append(name)


random.seed(42)
random.shuffle(sequences)

split = int(len(sequences) * 0.80)

test_sequences = sequences[split:]


print("=" * 60)
print("HPATCHES GROUND-TRUTH EVALUATION")
print("=" * 60)

print("Test sequences:", len(test_sequences))


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
# ORB
# ==================================================

orb = cv2.ORB_create()


# ==================================================
# FEATURES
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
# GET GOOD MATCHES
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


    good = []


    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:

            good.append(m)


    return good


# ==================================================
# HOMOGRAPHY ERROR
# ==================================================

def calculate_mma(
    kp1,
    kp2,
    matches,
    H
):

    if len(matches) == 0:

        return 0.0, 0.0, 0.0


    correct_1 = 0
    correct_3 = 0
    correct_5 = 0


    # Convert points from image 1
    # into homogeneous coordinates

    points = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ])


    points = points.reshape(
        -1,
        1,
        2
    )


    # Apply ground-truth homography

    projected = cv2.perspectiveTransform(
        points,
        H
    )


    for i, m in enumerate(matches):

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
            correct_1 += 1

        if error <= 3:
            correct_3 += 1

        if error <= 5:
            correct_5 += 1


    total = len(matches)


    return (
        correct_1 / total,
        correct_3 / total,
        correct_5 / total
    )


# ==================================================
# TOTALS
# ==================================================

orb_mma1 = []
orb_mma3 = []
orb_mma5 = []

rl_mma1 = []
rl_mma3 = []
rl_mma5 = []


# ==================================================
# EVALUATION
# ==================================================

print()
print("=" * 60)
print("STARTING GROUND-TRUTH TEST")
print("=" * 60)


for sequence in test_sequences:

    for image_number in range(2, 7):


        # ------------------------------------------
        # Paths
        # ------------------------------------------

        folder = os.path.join(
            DATASET,
            sequence
        )


        path1 = os.path.join(
            folder,
            "1.ppm"
        )


        path2 = os.path.join(
            folder,
            str(image_number) + ".ppm"
        )


        h_path = os.path.join(
            folder,
            "H_1_" + str(image_number)
        )


        # ------------------------------------------
        # Load images
        # ------------------------------------------

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
        # Load ground truth
        # ------------------------------------------

        try:

            H = np.loadtxt(
                h_path
            )

            H = H.reshape(
                3,
                3
            )

        except Exception:

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


        # ------------------------------------------
        # ORB matches
        # ------------------------------------------

        orb_matches = get_good_matches(
            des1,
            des2
        )


        (
            mma1,
            mma3,
            mma5
        ) = calculate_mma(
            kp1,
            kp2,
            orb_matches,
            H
        )


        orb_mma1.append(mma1)
        orb_mma3.append(mma3)
        orb_mma5.append(mma5)


        # ------------------------------------------
        # RL features
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
        # RL selection
        # ------------------------------------------

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


        # ------------------------------------------
        # RL matches
        # ------------------------------------------

        rl_matches = get_good_matches(
            rl_des1,
            rl_des2
        )


        (
            mma1,
            mma3,
            mma5
        ) = calculate_mma(
            rl_kp1,
            rl_kp2,
            rl_matches,
            H
        )


        rl_mma1.append(mma1)
        rl_mma3.append(mma3)
        rl_mma5.append(mma5)


# ==================================================
# FINAL MMA
# ==================================================

print()
print("=" * 60)
print("FINAL HPATCHES MMA RESULTS")
print("=" * 60)


print()
print("ORB")
print("--------------------------------")

print(
    "MMA @ 1px :",
    round(
        np.mean(orb_mma1) * 100,
        2
    ),
    "%"
)

print(
    "MMA @ 3px :",
    round(
        np.mean(orb_mma3) * 100,
        2
    ),
    "%"
)

print(
    "MMA @ 5px :",
    round(
        np.mean(orb_mma5) * 100,
        2
    ),
    "%"
)


print()
print("ORB + RL")
print("--------------------------------")

print(
    "MMA @ 1px :",
    round(
        np.mean(rl_mma1) * 100,
        2
    ),
    "%"
)

print(
    "MMA @ 3px :",
    round(
        np.mean(rl_mma3) * 100,
        2
    ),
    "%"
)

print(
    "MMA @ 5px :",
    round(
        np.mean(rl_mma5) * 100,
        2
    ),
    "%"
)


print()
print("Number of pairs:", len(orb_mma1))

print()
print("=" * 60)
print("HPATCHES EVALUATION COMPLETE")
print("=" * 60)
