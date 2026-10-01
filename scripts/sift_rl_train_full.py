import cv2
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ==================================================
# SETTINGS
# ==================================================

DATASET = "hpatches"

EPISODES = 2000

LEARNING_RATE = 0.00005

ENTROPY_WEIGHT = 0.05


# ==================================================
# RL POLICY
# ==================================================

class SiftPolicy(nn.Module):

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
print("SIFT + RL FULL-DATASET TRAINING")
print("=" * 60)

print(
    "Total sequences:",
    len(sequences)
)

print(
    "Available pairs:",
    len(sequences) * 5
)


# ==================================================
# SIFT
# ==================================================

sift = cv2.SIFT_create()


# ==================================================
# POLICY
# ==================================================

policy = SiftPolicy()

optimizer = optim.Adam(
    policy.parameters(),
    lr=LEARNING_RATE
)

running_baseline = 0.0


# ==================================================
# LOAD IMAGE PAIR + HOMOGRAPHY
# ==================================================

def load_pair(sequence, image_number):

    folder = os.path.join(
        DATASET,
        sequence
    )

    img1 = cv2.imread(
        os.path.join(
            folder,
            "1.ppm"
        ),
        cv2.IMREAD_GRAYSCALE
    )

    img2 = cv2.imread(
        os.path.join(
            folder,
            str(image_number) + ".ppm"
        ),
        cv2.IMREAD_GRAYSCALE
    )

    try:

        H = np.loadtxt(
            os.path.join(
                folder,
                "H_1_" + str(image_number)
            )
        ).reshape(3, 3)

    except Exception:

        H = None

    return img1, img2, H


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
        cv2.NORM_L2
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

        best_distance = 1000.0
        second_distance = 1000.0
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

            min(best_distance / 1000.0, 1.0),
            min(second_distance / 1000.0, 1.0),

            min(ratio, 1.0)
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

        selected_descriptors = np.asarray(
            selected_descriptors,
            dtype=np.float32
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
        cv2.NORM_L2
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
# RANSAC
# ==================================================

def get_inliers(
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


    return int(
        np.sum(mask)
    )


# ==================================================
# GROUND-TRUTH MMA
# ==================================================

def calculate_mma(
    kp1,
    kp2,
    matches,
    H
):

    if len(matches) == 0 or H is None:

        return 0.0, 0.0, 0.0


    points = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)


    projected = cv2.perspectiveTransform(
        points,
        H
    )


    c1 = 0
    c3 = 0
    c5 = 0


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
            c1 += 1

        if error <= 3:
            c3 += 1

        if error <= 5:
            c5 += 1


    total = len(matches)


    return (
        c1 / total,
        c3 / total,
        c5 / total
    )


# ==================================================
# TRAINING
# ==================================================

print()
print("=" * 60)
print("STARTING SIFT RL TRAINING")
print("=" * 60)


for episode in range(EPISODES):


    sequence = random.choice(
        sequences
    )


    image_number = random.randint(
        2,
        6
    )


    img1, img2, H = load_pair(
        sequence,
        image_number
    )


    if img1 is None or img2 is None or H is None:

        continue


    # ------------------------------------------
    # SIFT
    # ------------------------------------------

    kp1, des1 = sift.detectAndCompute(
        img1,
        None
    )

    kp2, des2 = sift.detectAndCompute(
        img2,
        None
    )


    if des1 is None or des2 is None:

        continue


    if len(kp1) == 0 or len(kp2) == 0:

        continue


    # ------------------------------------------
    # States
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


    # ------------------------------------------
    # Matching
    # ------------------------------------------

    good_matches = get_good_matches(
        selected_des1,
        selected_des2
    )


    # ------------------------------------------
    # RANSAC
    # ------------------------------------------

    inliers = get_inliers(
        selected_kp1,
        selected_kp2,
        good_matches
    )


    # ------------------------------------------
    # MMA
    # ------------------------------------------

    mma1, mma3, mma5 = calculate_mma(
        selected_kp1,
        selected_kp2,
        good_matches,
        H
    )


    # ------------------------------------------
    # Selection ratio
    # ------------------------------------------

    selected_count = (
        len(selected_kp1)
        +
        len(selected_kp2)
    )


    total_count = (
        len(kp1)
        +
        len(kp2)
    )


    selection_ratio = (
        selected_count /
        max(total_count, 1)
    )


    # ==================================================
    # ACCURACY-FOCUSED REWARD
    # ==================================================

    accuracy_reward = (
        2.0 * mma1
        +
        1.0 * mma3
        +
        0.5 * mma5
    )


    geometric_reward = (
        inliers / 50.0
    )


    efficiency_reward = (
        0.10 *
        (1.0 - selection_ratio)
    )


    reward = (
        accuracy_reward
        +
        geometric_reward
        +
        efficiency_reward
    )


    # Prevent complete collapse

    if selected_count < 50:

        reward -= 1.0


    # ==================================================
    # REINFORCE
    # ==================================================

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


    # ------------------------------------------
    # Baseline
    # ------------------------------------------

    if episode == 0:

        running_baseline = reward

    else:

        running_baseline = (
            0.95 * running_baseline
            +
            0.05 * reward
        )


    advantage = (
        reward -
        running_baseline
    )


    # ------------------------------------------
    # Loss
    # ------------------------------------------

    loss = (
        -advantage *
        log_prob_mean
        -
        ENTROPY_WEIGHT *
        entropy_mean
    )


    optimizer.zero_grad()

    loss.backward()


    torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        0.5
    )


    optimizer.step()


    # ------------------------------------------
    # Progress
    # ------------------------------------------

    if (
        episode == 0
        or (episode + 1) % 100 == 0
    ):

        print(
            "Episode:",
            episode + 1,
            "| Sequence:",
            sequence,
            "| Pair: 1->",
            image_number,
            "| Good:",
            len(good_matches),
            "| Inliers:",
            inliers,
            "| MMA1:",
            round(mma1 * 100, 2),
            "| MMA3:",
            round(mma3 * 100, 2),
            "| MMA5:",
            round(mma5 * 100, 2),
            "| Selected:",
            selected_count,
            "| Reward:",
            round(float(reward), 4)
        )


# ==================================================
# SAVE
# ==================================================

torch.save(
    policy.state_dict(),
    "scripts/sift_rl_policy_full.pth"
)


print()
print("=" * 60)
print("SIFT RL TRAINING COMPLETE")
print("=" * 60)

print(
    "Model saved:",
    "scripts/sift_rl_policy_full.pth"
)
