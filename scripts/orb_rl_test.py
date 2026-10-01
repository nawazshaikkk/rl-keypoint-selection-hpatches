import sys
import cv2
import torch
import torch.nn as nn
import numpy as np


# ==================================================
# CHECK COMMAND LINE ARGUMENT
# ==================================================

if len(sys.argv) < 2:
    print("Usage: python3 scripts/orb_rl_test.py 2")
    exit()

image_number = sys.argv[1]


# ==================================================
# RL POLICY
# ==================================================

class KeypointPolicy(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 2)
        )

    def forward(self, state):
        return self.network(state)


# ==================================================
# LOAD IMAGES
# ==================================================

img1 = cv2.imread(
    "hpatches/i_ajuntament/1.ppm",
    cv2.IMREAD_GRAYSCALE
)

img2 = cv2.imread(
    "hpatches/i_ajuntament/" + image_number + ".ppm",
    cv2.IMREAD_GRAYSCALE
)


# Check images

if img1 is None:
    print("Error: Could not load image 1")
    exit()

if img2 is None:
    print(
        "Error: Could not load image "
        + image_number
    )
    exit()


# ==================================================
# ORB
# ==================================================

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(
    img1,
    None
)

kp2, des2 = orb.detectAndCompute(
    img2,
    None
)


# ==================================================
# LOAD TRAINED RL MODEL
# ==================================================

policy = KeypointPolicy()

policy.load_state_dict(
    torch.load(
        "scripts/orb_rl_policy.pth",
        weights_only=True
    )
)

policy.eval()


# ==================================================
# RL KEYPOINT SELECTION
# ==================================================

def select_keypoints(
    keypoints,
    descriptors,
    policy,
    image_width,
    image_height
):

    selected_keypoints = []
    selected_descriptors = []

    for i, kp in enumerate(keypoints):

        # Create state
        state = torch.tensor([
            kp.pt[0] / image_width,
            kp.pt[1] / image_height,
            kp.size / 100.0,
            kp.angle / 360.0,
            kp.response
        ], dtype=torch.float32)

        state = state.unsqueeze(0)

        # Policy prediction
        output = policy(state)

        probability = torch.softmax(
            output,
            dim=1
        )

        # Deterministic action
        action = torch.argmax(
            probability,
            dim=1
        ).item()

        # 1 = KEEP
        if action == 1:

            selected_keypoints.append(kp)

            selected_descriptors.append(
                descriptors[i]
            )


    # Convert descriptors
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
# APPLY RL
# ==================================================

kp1_selected, des1_selected = select_keypoints(
    kp1,
    des1,
    policy,
    img1.shape[1],
    img1.shape[0]
)

kp2_selected, des2_selected = select_keypoints(
    kp2,
    des2,
    policy,
    img2.shape[1],
    img2.shape[0]
)


# ==================================================
# MATCHING
# ==================================================

good_matches = []
inlier_matches = []


if (
    des1_selected is not None
    and des2_selected is not None
    and len(des1_selected) >= 2
    and len(des2_selected) >= 2
):

    bf = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )

    raw_matches = bf.knnMatch(
        des1_selected,
        des2_selected,
        k=2
    )


    # Lowe ratio test

    for pair in raw_matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:

            good_matches.append(m)


    # ==================================================
    # RANSAC
    # ==================================================

    if len(good_matches) >= 4:

        src_pts = []
        dst_pts = []


        for m in good_matches:

            src_pts.append(
                kp1_selected[m.queryIdx].pt
            )

            dst_pts.append(
                kp2_selected[m.trainIdx].pt
            )


        src_pts = np.float32(
            src_pts
        ).reshape(-1, 1, 2)

        dst_pts = np.float32(
            dst_pts
        ).reshape(-1, 1, 2)


        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            5.0
        )


        if mask is not None:

            for i, m in enumerate(good_matches):

                if mask[i]:

                    inlier_matches.append(m)


# ==================================================
# SAVE VISUAL RESULT
# ==================================================

result = cv2.drawMatches(
    img1,
    kp1_selected,
    img2,
    kp2_selected,
    inlier_matches,
    None,
    flags=2
)

cv2.imwrite(
    "results/orb_rl_matches_" + image_number + ".png",
    result
)


# ==================================================
# PRINT RESULTS
# ==================================================

print()
print("=" * 50)
print("TRAINED ORB + RL RESULTS")
print("=" * 50)

print(
    "Image pair                 : 1 ->",
    image_number
)

print(
    "Original keypoints image 1 :",
    len(kp1)
)

print(
    "Original keypoints image 2 :",
    len(kp2)
)

print(
    "RL selected image 1        :",
    len(kp1_selected)
)

print(
    "RL selected image 2        :",
    len(kp2_selected)
)

print(
    "Good matches               :",
    len(good_matches)
)

print(
    "RANSAC inliers             :",
    len(inlier_matches)
)

print(
    "Saved : results/orb_rl_matches_"
    + image_number
    + ".png"
)
