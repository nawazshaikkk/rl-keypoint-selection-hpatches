import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

DATASET = "hpatches"
MODEL_PATH = "scripts/brisk_rl_policy_full.pth"
GRAPH_PATH = "results/brisk_before_after_accuracy_10k.png"

os.makedirs("results", exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

LOWE_RATIO = 0.75
RANSAC_THRESHOLD = 5.0

# Baseline BRISK results from the full 580-pair evaluation
BASELINE_MMA_1 = 45.52
BASELINE_MMA_3 = 77.63
BASELINE_MMA_5 = 84.11


# ============================================================
# RL POLICY NETWORK
# ============================================================

class RLPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device("cpu")

policy = RLPolicy().to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

policy.load_state_dict(state_dict)
policy.eval()

print("=" * 70)
print("BRISK + RL FULL DATASET EVALUATION")
print("=" * 70)
print("Model:", MODEL_PATH)


# ============================================================
# BRISK DETECTOR
# ============================================================

brisk = cv2.BRISK_create()


# ============================================================
# BUILD STATE FOR EACH KEYPOINT
# ============================================================

def build_states(keypoints, descriptors, other_descriptors, width, height):

    states = []

    if descriptors is None or other_descriptors is None:
        return np.empty((0, 8), dtype=np.float32)

    # BF matcher for descriptor distances
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    knn_matches = bf.knnMatch(
        descriptors,
        other_descriptors,
        k=2
    )

    for i, kp in enumerate(keypoints):

        x = kp.pt[0] / width
        y = kp.pt[1] / height

        size = kp.size / 100.0

        angle = kp.angle / 360.0

        response = kp.response

        best_distance = 0.0
        second_distance = 0.0
        ratio = 1.0

        if i < len(knn_matches) and len(knn_matches[i]) >= 2:

            m = knn_matches[i][0]
            n = knn_matches[i][1]

            best_distance = m.distance / 256.0
            second_distance = n.distance / 256.0

            if n.distance > 0:
                ratio = m.distance / n.distance

        state = [
            x,
            y,
            size,
            angle,
            response,
            best_distance,
            second_distance,
            ratio
        ]

        states.append(state)

    return np.asarray(states, dtype=np.float32)


# ============================================================
# RL KEYPOINT SELECTION
# ============================================================

def select_keypoints(keypoints, descriptors, states):

    if len(keypoints) == 0:
        return [], None

    if len(states) == 0:
        return [], None

    with torch.no_grad():

        x = torch.tensor(
            states,
            dtype=torch.float32,
            device=device
        )

        logits = policy(x)

        actions = torch.argmax(logits, dim=1).cpu().numpy()

    keep_indices = np.where(actions == 1)[0]

    selected_keypoints = [
        keypoints[i]
        for i in keep_indices
    ]

    if descriptors is not None and len(keep_indices) > 0:
        selected_descriptors = descriptors[keep_indices]
    else:
        selected_descriptors = None

    return selected_keypoints, selected_descriptors


# ============================================================
# LOWE MATCHING
# ============================================================

def lowe_matches(des1, des2):

    if des1 is None or des2 is None:
        return []

    if len(des1) < 2 or len(des2) < 2:
        return []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    knn = bf.knnMatch(
        des1,
        des2,
        k=2
    )

    good = []

    for pair in knn:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)

    return good


# ============================================================
# RANSAC
# ============================================================

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
            RANSAC_THRESHOLD
        )

        if mask is None:
            return 0

        return int(mask.sum())

    except:
        return 0


# ============================================================
# MMA CALCULATION
# ============================================================

def compute_mma(kp1, kp2, matches, H):

    if H is None:
        return 0.0, 0.0, 0.0

    if len(matches) == 0:
        return 0.0, 0.0, 0.0

    errors = []

    for m in matches:

        p1 = np.array(
            [
                [
                    kp1[m.queryIdx].pt
                ]
            ],
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


# ============================================================
# GET ALL HPATCHES SEQUENCES
# ============================================================

sequences = sorted([
    s for s in os.listdir(DATASET)
    if os.path.isdir(os.path.join(DATASET, s))
])

print("Total sequences:", len(sequences))
print("Expected pairs:", len(sequences) * 5)

# ============================================================
# ACCUMULATORS
# ============================================================

all_mma1 = []
all_mma3 = []
all_mma5 = []

all_keypoints = []
all_good_matches = []
all_inliers = []

total_pairs = 0


# ============================================================
# PROCESS ALL 580 PAIRS
# ============================================================

for seq_idx, sequence in enumerate(sequences, start=1):

    seq_path = os.path.join(
        DATASET,
        sequence
    )

    img1_path = os.path.join(
        seq_path,
        "1.ppm"
    )

    img1 = cv2.imread(
        img1_path,
        cv2.IMREAD_GRAYSCALE
    )

    if img1 is None:
        print("Could not read:", img1_path)
        continue

    # --------------------------------------------------------
    # Reference image keypoints/descriptors
    # --------------------------------------------------------

    kp1_all, des1_all = brisk.detectAndCompute(
        img1,
        None
    )

    if des1_all is None or len(kp1_all) == 0:
        continue

    h1, w1 = img1.shape

    for image_number in range(2, 7):

        img2_path = os.path.join(
            seq_path,
            f"{image_number}.ppm"
        )

        H_path = os.path.join(
            seq_path,
            f"H_1_{image_number}"
        )

        img2 = cv2.imread(
            img2_path,
            cv2.IMREAD_GRAYSCALE
        )

        H = np.loadtxt(H_path)

        if img2 is None:
            continue

        # ----------------------------------------------------
        # Detect BRISK features in image 2
        # ----------------------------------------------------

        kp2_all, des2_all = brisk.detectAndCompute(
            img2,
            None
        )

        if des2_all is None or len(kp2_all) == 0:
            continue

        h2, w2 = img2.shape

        # ----------------------------------------------------
        # Build RL states
        # ----------------------------------------------------

        states1 = build_states(
            kp1_all,
            des1_all,
            des2_all,
            w1,
            h1
        )

        states2 = build_states(
            kp2_all,
            des2_all,
            des1_all,
            w2,
            h2
        )

        # ----------------------------------------------------
        # RL selection
        # ----------------------------------------------------

        kp1_selected, des1_selected = select_keypoints(
            kp1_all,
            des1_all,
            states1
        )

        kp2_selected, des2_selected = select_keypoints(
            kp2_all,
            des2_all,
            states2
        )

        # ----------------------------------------------------
        # Match selected descriptors
        # ----------------------------------------------------

        matches = lowe_matches(
            des1_selected,
            des2_selected
        )

        # ----------------------------------------------------
        # RANSAC
        # ----------------------------------------------------

        inliers = compute_ransac_inliers(
            kp1_selected,
            kp2_selected,
            matches
        )

        # ----------------------------------------------------
        # MMA
        # ----------------------------------------------------

        mma1, mma3, mma5 = compute_mma(
            kp1_selected,
            kp2_selected,
            matches,
            H
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        all_mma1.append(mma1)
        all_mma3.append(mma3)
        all_mma5.append(mma5)

        all_keypoints.append(
            (
                len(kp1_selected)
                +
                len(kp2_selected)
            ) / 2.0
        )

        all_good_matches.append(
            len(matches)
        )

        all_inliers.append(
            inliers
        )

        total_pairs += 1

        if total_pairs % 20 == 0:

            print(
                f"[{total_pairs}/580] "
                f"{sequence} 1->{image_number} | "
                f"Selected: "
                f"{(len(kp1_selected)+len(kp2_selected))/2:.0f} | "
                f"Good: {len(matches)} | "
                f"Inliers: {inliers} | "
                f"MMA@1: {mma1:.2f}% | "
                f"MMA@3: {mma3:.2f}% | "
                f"MMA@5: {mma5:.2f}%"
            )


# ============================================================
# FINAL RESULTS
# ============================================================

avg_mma1 = np.mean(all_mma1)
avg_mma3 = np.mean(all_mma3)
avg_mma5 = np.mean(all_mma5)

avg_keypoints = np.mean(all_keypoints)
avg_good_matches = np.mean(all_good_matches)
avg_inliers = np.mean(all_inliers)


print()
print("=" * 70)
print("BRISK + RL FULL DATASET RESULTS")
print("=" * 70)

print(f"Total evaluated pairs : {total_pairs}")

print(f"Average selected keypoints : {avg_keypoints:.2f}")
print(f"Average good matches       : {avg_good_matches:.2f}")
print(f"Average RANSAC inliers     : {avg_inliers:.2f}")

print()
print(f"MMA @ 1 px : {avg_mma1:.2f}%")
print(f"MMA @ 3 px : {avg_mma3:.2f}%")
print(f"MMA @ 5 px : {avg_mma5:.2f}%")

print()
print("=" * 70)
print("BASELINE vs BRISK + RL")
print("=" * 70)

print(f"MMA @ 1 px : {BASELINE_MMA_1:.2f}% -> {avg_mma1:.2f}%")
print(f"MMA @ 3 px : {BASELINE_MMA_3:.2f}% -> {avg_mma3:.2f}%")
print(f"MMA @ 5 px : {BASELINE_MMA_5:.2f}% -> {avg_mma5:.2f}%")


# ============================================================
# ACCURACY GRAPH
# ============================================================

metrics = [
    "MMA @ 1 px",
    "MMA @ 3 px",
    "MMA @ 5 px"
]

baseline = [
    BASELINE_MMA_1,
    BASELINE_MMA_3,
    BASELINE_MMA_5
]

rl = [
    avg_mma1,
    avg_mma3,
    avg_mma5
]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(9, 6))

plt.bar(
    x - width / 2,
    baseline,
    width,
    label="BRISK"
)

plt.bar(
    x + width / 2,
    rl,
    width,
    label="BRISK + RL"
)

plt.xticks(
    x,
    metrics
)

plt.ylabel("Matching Accuracy (%)")
plt.xlabel("MMA Threshold")

plt.title(
    "BRISK vs BRISK + RL Accuracy\n"
    "HPatches Full Dataset (580 Pairs)"
)

plt.ylim(0, 100)

plt.legend()

plt.tight_layout()

plt.savefig(
    GRAPH_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Accuracy graph saved to:")
print(GRAPH_PATH)

print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
