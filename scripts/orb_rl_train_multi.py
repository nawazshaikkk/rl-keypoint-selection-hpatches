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

EPISODES = 10000

LEARNING_RATE = 0.0001

ENTROPY_WEIGHT = 0.05

SELECTION_PENALTY = 0.20


# ==================================================
# RL POLICY NETWORK
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
# FIND ALL HPATCHES SEQUENCES
# ==================================================

sequences = []

for name in sorted(os.listdir(DATASET)):

    path = os.path.join(
        DATASET,
        name
    )

    if os.path.isdir(path):

        sequences.append(name)


if len(sequences) < 1:

    print(
        "No HPatches sequences found."
    )

    exit()


# ==================================================
# USE COMPLETE DATASET
# ==================================================

train_sequences = sequences

total_available_pairs = (
    len(train_sequences) * 5
)


print("=" * 60)
print("HPATCHES FULL-DATASET RL TRAINING")
print("=" * 60)

print(
    "Total sequences :",
    len(sequences)
)

print(
    "Training        :",
    len(train_sequences)
)

print(
    "Available pairs :",
    total_available_pairs
)

print()
print(
    "Training uses ALL HPatches sequences."
)

print(
    "Pairs per sequence: 1->2, 1->3, 1->4, 1->5, 1->6"
)

print(
    "Total available pairs:",
    len(train_sequences),
    "x 5 =",
    total_available_pairs
)


# ==================================================
# ORB
# ==================================================

orb = cv2.ORB_create()


# ==================================================
# POLICY
# ==================================================

policy = KeypointPolicy()

optimizer = optim.Adam(
    policy.parameters(),
    lr=LEARNING_RATE
)


# ==================================================
# RUNNING REWARD BASELINE
# ==================================================

running_baseline = 0.0


# ==================================================
# LOAD IMAGE PAIR
# ==================================================

def load_pair(
    sequence,
    image_number
):

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


    return img1, img2


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
        and
        other_descriptors is not None
    ):

        try:

            matches = bf.knnMatch(
                descriptors,
                other_descriptors,
                k=2
            )

        except cv2.error:

            matches = []

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
                        m.distance
                        /
                        n.distance
                    )


        state = [

            # Position

            kp.pt[0] / width,

            kp.pt[1] / height,


            # ORB properties

            kp.size / 100.0,

            kp.angle / 360.0,

            kp.response,


            # Descriptor information

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
# SELECT KEYPOINTS
# ==================================================

def select_keypoints(
    keypoints,
    descriptors,
    states,
    policy
):

    selected_keypoints = []

    selected_descriptors = []

    log_probs = []

    entropies = []


    for i in range(
        len(keypoints)
    ):


        state = states[i].unsqueeze(0)


        logits = policy(
            state
        )


        probabilities = torch.softmax(
            logits,
            dim=1
        )


        distribution = (
            torch.distributions.Categorical(
                probabilities
            )
        )


        action = distribution.sample()


        log_probs.append(
            distribution.log_prob(
                action
            )
        )


        entropies.append(
            distribution.entropy()
        )


        # ------------------------------------------
        # ACTION
        #
        # 0 = REJECT
        # 1 = KEEP
        # ------------------------------------------

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
# MATCHING + RANSAC
# ==================================================

def calculate_results(
    kp1,
    des1,
    kp2,
    des2
):


    if (
        des1 is None
        or
        des2 is None
    ):

        return 0, 0


    if (
        len(des1) < 2
        or
        len(des2) < 2
    ):

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


    # ==================================================
    # LOWE RATIO TEST
    # ==================================================

    for pair in matches:

        if len(pair) < 2:

            continue


        m, n = pair


        if m.distance < 0.75 * n.distance:

            good_matches.append(m)


    if len(good_matches) < 4:

        return (
            len(good_matches),
            0
        )


    # ==================================================
    # RANSAC
    # ==================================================

    src_pts = np.float32([

        kp1[m.queryIdx].pt

        for m in good_matches

    ]).reshape(
        -1,
        1,
        2
    )


    dst_pts = np.float32([

        kp2[m.trainIdx].pt

        for m in good_matches

    ]).reshape(
        -1,
        1,
        2
    )


    try:

        H, mask = cv2.findHomography(

            src_pts,

            dst_pts,

            cv2.RANSAC,

            5.0
        )

    except cv2.error:

        return (
            len(good_matches),
            0
        )


    if mask is None:

        return (
            len(good_matches),
            0
        )


    inliers = int(
        np.sum(mask)
    )


    return (

        len(good_matches),

        inliers
    )


# ==================================================
# TRAINING
# ==================================================

print()
print("=" * 60)
print("STARTING FULL-DATASET RL TRAINING")
print("=" * 60)

print(
    "Episodes:",
    EPISODES
)

print(
    "Training sequences:",
    len(train_sequences)
)

print(
    "Available image pairs:",
    total_available_pairs
)


# ==================================================
# EPISODES
# ==================================================

for episode in range(
    EPISODES
):


    # ------------------------------------------
    # RANDOM SEQUENCE FROM ALL 116
    # ------------------------------------------

    sequence = random.choice(
        train_sequences
    )


    # ------------------------------------------
    # RANDOM TARGET IMAGE
    #
    # 2,3,4,5,6
    # ------------------------------------------

    image_number = random.randint(
        2,
        6
    )


    # ------------------------------------------
    # LOAD IMAGES
    # ------------------------------------------

    img1, img2 = load_pair(
        sequence,
        image_number
    )


    if (
        img1 is None
        or
        img2 is None
    ):

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


    if (
        des1 is None
        or
        des2 is None
    ):

        continue


    if (
        len(kp1) == 0
        or
        len(kp2) == 0
    ):

        continue


    # ==================================================
    # CREATE STATES
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
    # RL SELECTION IMAGE 1
    # ==================================================

    (

        selected_kp1,

        selected_des1,

        log_probs1,

        entropy1

    ) = select_keypoints(

        kp1,

        des1,

        states1,

        policy
    )


    # ==================================================
    # RL SELECTION IMAGE 2
    # ==================================================

    (

        selected_kp2,

        selected_des2,

        log_probs2,

        entropy2

    ) = select_keypoints(

        kp2,

        des2,

        states2,

        policy
    )


    # ==================================================
    # MATCHING
    # ==================================================

    good_matches, inliers = calculate_results(

        selected_kp1,

        selected_des1,

        selected_kp2,

        selected_des2
    )


    # ==================================================
    # KEYPOINT COUNT
    # ==================================================

    selected_count = (

        len(selected_kp1)

        +

        len(selected_kp2)
    )


    total_keypoints = (

        len(kp1)

        +

        len(kp2)
    )


    # ==================================================
    # REWARD
    # ==================================================

    # Main reward:
    # geometrically valid RANSAC inliers

    matching_reward = (

        inliers

        /

        100.0
    )


    # ------------------------------------------
    # Efficiency reward
    # ------------------------------------------

    selection_ratio = (

        selected_count

        /

        max(
            total_keypoints,
            1
        )
    )


    efficiency_reward = (

        1.0

        -

        selection_ratio
    )


    # ------------------------------------------
    # Combined reward
    # ------------------------------------------

    reward = (

        matching_reward

        +

        SELECTION_PENALTY

        *

        efficiency_reward
    )


    # ------------------------------------------
    # Prevent zero selection
    # ------------------------------------------

    if selected_count < 20:

        reward -= 1.0


    # ==================================================
    # REINFORCE
    # ==================================================

    all_log_probs = (

        log_probs1

        +

        log_probs2
    )


    all_entropies = (

        entropy1

        +

        entropy2
    )


    if len(all_log_probs) == 0:

        continue


    # IMPORTANT:
    #
    # MEAN instead of SUM
    #
    # Prevents huge gradients.

    log_prob_mean = torch.stack(
        all_log_probs
    ).mean()


    entropy_mean = torch.stack(
        all_entropies
    ).mean()


    # ==================================================
    # RUNNING BASELINE
    # ==================================================

    if episode == 0:

        running_baseline = reward

    else:

        running_baseline = (

            0.95

            *

            running_baseline

            +

            0.05

            *

            reward
        )


    # ==================================================
    # ADVANTAGE
    # ==================================================

    advantage = (

        reward

        -

        running_baseline
    )


    # ==================================================
    # LOSS
    # ==================================================

    loss = (

        -advantage

        *

        log_prob_mean

        -

        ENTROPY_WEIGHT

        *

        entropy_mean
    )


    # ==================================================
    # UPDATE POLICY
    # ==================================================

    optimizer.zero_grad()


    loss.backward()


    torch.nn.utils.clip_grad_norm_(

        policy.parameters(),

        0.5
    )


    optimizer.step()


    # ==================================================
    # PRINT PROGRESS
    # ==================================================

    if (

        episode == 0

        or

        (episode + 1) % 100 == 0

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

            round(
                float(reward),
                4
            )
        )


# ==================================================
# SAVE NEW MODEL
# ==================================================

MODEL_PATH = (
    "scripts/orb_rl_policy_10000_full.pth"
)


torch.save(

    policy.state_dict(),

    MODEL_PATH
)


# ==================================================
# COMPLETE
# ==================================================

print()
print("=" * 60)
print("FULL-DATASET RL TRAINING COMPLETE")
print("=" * 60)

print(
    "Total sequences used:",
    len(train_sequences)
)

print(
    "Available image pairs:",
    total_available_pairs
)

print(
    "Training episodes:",
    EPISODES
)

print(
    "Model saved:",
    MODEL_PATH
)
