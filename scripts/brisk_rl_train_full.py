import cv2
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DATASET = "hpatches"
EPISODES = 10000
LEARNING_RATE = 0.0001
ENTROPY_WEIGHT = 0.05
SELECTION_PENALTY = 0.20


# ==================================================
# RL POLICY
# ==================================================

class BriskPolicy(nn.Module):

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
# HPATCHES SEQUENCES
# ==================================================

sequences = sorted([
    name for name in os.listdir(DATASET)
    if os.path.isdir(os.path.join(DATASET, name))
])

print("=" * 60)
print("BRISK + RL FULL-DATASET TRAINING")
print("=" * 60)

print("Total sequences:", len(sequences))
print("Training pairs available:", len(sequences) * 5)


# ==================================================
# BRISK
# ==================================================

brisk = cv2.BRISK_create()


# ==================================================
# POLICY
# ==================================================

policy = BriskPolicy()

optimizer = optim.Adam(
    policy.parameters(),
    lr=LEARNING_RATE
)

running_baseline = 0.0


# ==================================================
# LOAD PAIR
# ==================================================

def load_pair(sequence, image_number):

    folder = os.path.join(
        DATASET,
        sequence
    )

    img1 = cv2.imread(
        os.path.join(folder, "1.ppm"),
        cv2.IMREAD_GRAYSCALE
    )

    img2 = cv2.imread(
        os.path.join(folder, str(image_number) + ".ppm"),
        cv2.IMREAD_GRAYSCALE
    )

    return img1, img2


# ==================================================
# CREATE 8-DIMENSIONAL STATE
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

    log_probs = []
    entropies = []

    for i in range(len(keypoints)):

        state = states[i].unsqueeze(0)

        logits = policy(state)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        distribution = torch.distributions.Categorical(
            probabilities
        )

        action = distribution.sample()

        log_probs.append(
            distribution.log_prob(action)
        )

        entropies.append(
            distribution.entropy()
        )

        # 0 = REJECT
        # 1 = KEEP

        if action.item() == 1:

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
        selected_descriptors,
        log_probs,
        entropies
    )


# ==================================================
# MATCH + RANSAC
# ==================================================

def calculate_results(
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

        _, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            5.0
        )

    except cv2.error:

        return len(good_matches), 0

    if mask is None:
        return len(good_matches), 0

    inliers = int(np.sum(mask))

    return (
        len(good_matches),
        inliers
    )


# ==================================================
# TRAINING
# ==================================================

print()
print("=" * 60)
print("STARTING BRISK RL TRAINING")
print("=" * 60)

for episode in range(EPISODES):

    sequence = random.choice(sequences)

    image_number = random.randint(2, 6)

    img1, img2 = load_pair(
        sequence,
        image_number
    )

    if img1 is None or img2 is None:
        continue

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

    if len(kp1) == 0 or len(kp2) == 0:
        continue

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

    (
        selected_kp1,
        selected_des1,
        log_probs1,
        entropy1
    ) = select_keypoints(
        kp1,
        des1,
        states1
    )

    (
        selected_kp2,
        selected_des2,
        log_probs2,
        entropy2
    ) = select_keypoints(
        kp2,
        des2,
        states2
    )

    good_matches, inliers = calculate_results(
        selected_kp1,
        selected_des1,
        selected_kp2,
        selected_des2
    )

    selected_count = (
        len(selected_kp1) +
        len(selected_kp2)
    )

    total_keypoints = (
        len(kp1) +
        len(kp2)
    )

    # ------------------------------------------
    # Reward
    # ------------------------------------------

    matching_reward = (
        inliers / 100.0
    )

    selection_ratio = (
        selected_count /
        max(total_keypoints, 1)
    )

    efficiency_reward = (
        1.0 -
        selection_ratio
    )

    reward = (
        matching_reward +
        SELECTION_PENALTY *
        efficiency_reward
    )

    if selected_count < 20:
        reward -= 1.0

    # ------------------------------------------
    # REINFORCE
    # ------------------------------------------

    all_log_probs = (
        log_probs1 +
        log_probs2
    )

    all_entropies = (
        entropy1 +
        entropy2
    )

    if len(all_log_probs) == 0:
        continue

    log_prob_mean = torch.stack(
        all_log_probs
    ).mean()

    entropy_mean = torch.stack(
        all_entropies
    ).mean()

    if episode == 0:
        running_baseline = reward
    else:
        running_baseline = (
            0.95 * running_baseline +
            0.05 * reward
        )

    advantage = (
        reward -
        running_baseline
    )

    loss = (
        -advantage * log_prob_mean
        -
        ENTROPY_WEIGHT * entropy_mean
    )

    optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        0.5
    )

    optimizer.step()

    if (
        episode == 0
        or (episode + 1) % 50 == 0
    ):

        print(
            "Episode:",
            episode + 1,
            "| Sequence:",
            sequence,
            "| Pair: 1->",
            image_number,
            "| Good:",
            good_matches,
            "| Inliers:",
            inliers,
            "| Selected:",
            selected_count,
            "| Reward:",
            round(float(reward), 4)
        )


# ==================================================
# SAVE MODEL
# ==================================================

torch.save(
    policy.state_dict(),
    "scripts/brisk_rl_policy_10000_full.pth"
)

print()
print("=" * 60)
print("BRISK RL TRAINING COMPLETE")
print("=" * 60)

print(
    "Model saved:",
    "scripts/brisk_rl_policy_full.pth"
)
