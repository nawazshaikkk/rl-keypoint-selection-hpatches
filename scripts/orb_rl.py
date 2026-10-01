import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


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
    "hpatches/i_ajuntament/2.ppm",
    cv2.IMREAD_GRAYSCALE
)

if img1 is None or img2 is None:
    print("Error loading images.")
    exit()


# ==================================================
# ORB
# ==================================================

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

print("Original keypoints")
print("Image 1:", len(kp1))
print("Image 2:", len(kp2))


# ==================================================
# POLICY + OPTIMIZER
# ==================================================

policy = KeypointPolicy()

optimizer = optim.Adam(
    policy.parameters(),
    lr=0.0005
)


# ==================================================
# SELECT KEYPOINTS
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

    log_probs = []
    entropies = []

    for i, kp in enumerate(keypoints):

        state = torch.tensor([
            kp.pt[0] / image_width,
            kp.pt[1] / image_height,
            kp.size / 100.0,
            kp.angle / 360.0,
            kp.response
        ], dtype=torch.float32)

        state = state.unsqueeze(0)

        logits = policy(state)

        probability = torch.softmax(
            logits,
            dim=1
        )

        distribution = torch.distributions.Categorical(
            probability
        )

        action = distribution.sample()

        log_prob = distribution.log_prob(action)

        entropy = distribution.entropy()

        log_probs.append(log_prob)
        entropies.append(entropy)

        # 1 = KEEP
        if action.item() == 1:

            selected_keypoints.append(kp)

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
# MATCHING + RANSAC
# ==================================================

def calculate_results(
    kp1_selected,
    des1_selected,
    kp2_selected,
    des2_selected
):

    if (
        des1_selected is None
        or des2_selected is None
    ):
        return 0, 0

    if (
        len(des1_selected) < 2
        or len(des2_selected) < 2
    ):
        return 0, 0

    bf = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )

    try:

        raw_matches = bf.knnMatch(
            des1_selected,
            des2_selected,
            k=2
        )

    except cv2.error:

        return 0, 0

    good_matches = []

    for pair in raw_matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return len(good_matches), 0

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

    inliers = int(np.sum(mask))

    return len(good_matches), inliers


# ==================================================
# TRAINING
# ==================================================

episodes = 500

print()
print("=" * 45)
print("STARTING STABLE RL TRAINING")
print("=" * 45)


for episode in range(episodes):

    # ----------------------------------------------
    # Select image 1 keypoints
    # ----------------------------------------------

    (
        kp1_selected,
        des1_selected,
        log_probs1,
        entropies1
    ) = select_keypoints(
        kp1,
        des1,
        policy,
        img1.shape[1],
        img1.shape[0]
    )


    # ----------------------------------------------
    # Select image 2 keypoints
    # ----------------------------------------------

    (
        kp2_selected,
        des2_selected,
        log_probs2,
        entropies2
    ) = select_keypoints(
        kp2,
        des2,
        policy,
        img2.shape[1],
        img2.shape[0]
    )


    # ----------------------------------------------
    # Matching
    # ----------------------------------------------

    good_matches, inliers = calculate_results(
        kp1_selected,
        des1_selected,
        kp2_selected,
        des2_selected
    )


    selected_count = (
        len(kp1_selected)
        + len(kp2_selected)
    )


    # ==================================================
    # REWARD
    # ==================================================

    total_points = len(kp1) + len(kp2)

    selection_ratio = (
        selected_count / total_points
    )

    # Main matching reward
    matching_reward = inliers / 226.0

    # Reward keeping fewer points
    efficiency_reward = 0.3 * (
        1.0 - selection_ratio
    )

    # Penalize selecting almost nothing
    if selected_count < 100:

        minimum_points_penalty = -1.0

    else:

        minimum_points_penalty = 0.0


    reward = (
        matching_reward
        + efficiency_reward
        + minimum_points_penalty
    )


    # ==================================================
    # REINFORCE
    # ==================================================

    all_log_probs = (
        log_probs1 +
        log_probs2
    )

    all_entropies = (
        entropies1 +
        entropies2
    )


    log_prob_sum = torch.stack(
        all_log_probs
    ).sum()


    entropy_mean = torch.stack(
        all_entropies
    ).mean()


    # Entropy prevents policy collapse
    entropy_weight = 0.01


    loss = (
        -reward * log_prob_sum
        - entropy_weight * entropy_mean
    )


    # ----------------------------------------------
    # Update
    # ----------------------------------------------

    optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        1.0
    )

    optimizer.step()


    # ----------------------------------------------
    # Print
    # ----------------------------------------------

    if (
        episode == 0
        or (episode + 1) % 25 == 0
    ):

        print(
            "Episode:",
            episode + 1,
            "| Good:",
            good_matches,
            "| Inliers:",
            inliers,
            "| Selected:",
            selected_count,
            "| Reward:",
            round(reward, 4)
        )


# ==================================================
# SAVE MODEL
# ==================================================

torch.save(
    policy.state_dict(),
    "scripts/orb_rl_policy.pth"
)


print()
print("=" * 45)
print("TRAINING COMPLETE")
print("=" * 45)

print(
    "Model saved:",
    "scripts/orb_rl_policy.pth"
)
