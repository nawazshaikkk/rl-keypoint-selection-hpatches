import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# PATHS
# ============================================================

DATASET_ROOT = "../hpatches"

ORB_MODEL = "orb_rl_policy_10000_full.pth"
BRISK_MODEL = "brisk_rl_policy_10000_full.pth"


# ============================================================
# RL POLICY
# ============================================================

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

    def forward(self, x):
        return self.network(x)


# ============================================================
# LOAD POLICY
# ============================================================

def load_policy(path):
    model = KeypointPolicy()

    state = torch.load(path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()

    return model


# ============================================================
# HPATCHES SEQUENCES
# ============================================================

def get_sequences():
    sequences = []

    for name in sorted(os.listdir(DATASET_ROOT)):
        path = os.path.join(DATASET_ROOT, name)

        if os.path.isdir(path):
            # HPatches:
            # i_* = illumination
            # v_* = viewpoint
            if name.startswith("i_") or name.startswith("v_"):
                sequences.append(name)

    return sequences


# ============================================================
# DETECTOR
# ============================================================

def detect_and_compute(img, detector_name):

    if detector_name == "ORB":
        detector = cv2.ORB_create()
    else:
        detector = cv2.BRISK_create()

    keypoints, descriptors = detector.detectAndCompute(img, None)

    return keypoints, descriptors


# ============================================================
# DESCRIPTOR MATCHER
# ============================================================

def get_matcher(detector_name):

    if detector_name in ["ORB", "BRISK"]:
        return cv2.BFMatcher(cv2.NORM_HAMMING)

    raise ValueError("Unknown detector")


# ============================================================
# GET DESCRIPTOR DISTANCE FEATURES
# ============================================================

def descriptor_distance_features(desc1, desc2, detector_name):

    if desc1 is None or desc2 is None:
        return None

    if len(desc1) == 0 or len(desc2) == 0:
        return None

    matcher = get_matcher(detector_name)

    matches = matcher.knnMatch(desc1, desc2, k=2)

    best = np.full(len(desc1), 999999.0, dtype=np.float32)
    second = np.full(len(desc1), 999999.0, dtype=np.float32)

    for i, pair in enumerate(matches):

        if len(pair) >= 1:
            best[i] = pair[0].distance

        if len(pair) >= 2:
            second[i] = pair[1].distance

    ratio = best / (second + 1e-8)

    return best, second, ratio


# ============================================================
# BUILD 8-D STATE
# ============================================================

def build_state(keypoints, distance_features, width, height):

    best, second, ratio = distance_features

    states = []

    for i, kp in enumerate(keypoints):

        x = kp.pt[0] / width
        y = kp.pt[1] / height

        size = kp.size
        angle = kp.angle

        if angle < 0:
            angle = 0.0

        angle = angle / 360.0

        response = kp.response

        # Safe normalization
        size = size / 100.0
        response = float(response)

        state = [
            x,
            y,
            size,
            angle,
            response,
            best[i] / 256.0,
            second[i] / 256.0,
            ratio[i]
        ]

        states.append(state)

    if len(states) == 0:
        return torch.empty((0, 8), dtype=torch.float32)

    return torch.tensor(states, dtype=torch.float32)


# ============================================================
# RL FILTER
# ============================================================

def apply_rl_filter(
    keypoints,
    descriptors,
    desc_features,
    policy,
    width,
    height
):

    if len(keypoints) == 0:
        return [], None

    states = build_state(
        keypoints,
        desc_features,
        width,
        height
    )

    with torch.no_grad():
        logits = policy(states)

        # 0 = REJECT
        # 1 = KEEP
        actions = torch.argmax(logits, dim=1).numpy()

    selected_indices = np.where(actions == 1)[0]

    selected_keypoints = [
        keypoints[i] for i in selected_indices
    ]

    if descriptors is not None:
        selected_descriptors = descriptors[selected_indices]
    else:
        selected_descriptors = None

    return selected_keypoints, selected_descriptors


# ============================================================
# LOWE RATIO MATCHING
# ============================================================

def get_good_matches(desc1, desc2, detector_name):

    if desc1 is None or desc2 is None:
        return []

    if len(desc1) == 0 or len(desc2) == 0:
        return []

    matcher = get_matcher(detector_name)

    knn_matches = matcher.knnMatch(desc1, desc2, k=2)

    good = []

    for pair in knn_matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good.append(m)

    return good


# ============================================================
# RANSAC
# ============================================================

def get_inliers(kp1, kp2, matches):

    if len(matches) < 4:
        return 0

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        5.0
    )

    if mask is None:
        return 0

    return int(mask.sum())


# ============================================================
# MMA USING HPATCHES GROUND TRUTH
# ============================================================

def compute_mma(kp1, kp2, matches, H_gt):

    if len(matches) == 0:
        return [0.0, 0.0, 0.0]

    errors = []

    for m in matches:

        pt1 = np.array(
            [kp1[m.queryIdx].pt[0],
             kp1[m.queryIdx].pt[1],
             1.0],
            dtype=np.float64
        )

        projected = H_gt @ pt1

        if abs(projected[2]) < 1e-12:
            continue

        projected = projected / projected[2]

        pt2 = np.array(
            kp2[m.trainIdx].pt,
            dtype=np.float64
        )

        error = np.linalg.norm(
            projected[:2] - pt2
        )

        errors.append(error)

    if len(errors) == 0:
        return [0.0, 0.0, 0.0]

    errors = np.array(errors)

    mma1 = np.mean(errors <= 1.0) * 100
    mma3 = np.mean(errors <= 3.0) * 100
    mma5 = np.mean(errors <= 5.0) * 100

    return [mma1, mma3, mma5]


# ============================================================
# READ IMAGE + HOMOGRAPHY
# ============================================================

def read_pair(sequence, pair_number):

    seq_path = os.path.join(
        DATASET_ROOT,
        sequence
    )

    img1_path = os.path.join(
        seq_path,
        "1.ppm"
    )

    img2_path = os.path.join(
        seq_path,
        f"{pair_number}.ppm"
    )

    H_path = os.path.join(
        seq_path,
        f"H_1_{pair_number}"
    )

    img1 = cv2.imread(
        img1_path,
        cv2.IMREAD_GRAYSCALE
    )

    img2 = cv2.imread(
        img2_path,
        cv2.IMREAD_GRAYSCALE
    )

    H_gt = np.loadtxt(H_path)

    return img1, img2, H_gt


# ============================================================
# EVALUATE ONE PAIR
# ============================================================

def evaluate_pair(
    img1,
    img2,
    H_gt,
    detector_name,
    policy=None
):

    # --------------------------------------------------------
    # Detection + descriptor extraction
    # --------------------------------------------------------

    kp1, desc1 = detect_and_compute(
        img1,
        detector_name
    )

    kp2, desc2 = detect_and_compute(
        img2,
        detector_name
    )

    # --------------------------------------------------------
    # RL filtering
    # --------------------------------------------------------

    if policy is not None:

        feat1 = descriptor_distance_features(
            desc1,
            desc2,
            detector_name
        )

        feat2 = descriptor_distance_features(
            desc2,
            desc1,
            detector_name
        )

        if feat1 is not None:
            kp1, desc1 = apply_rl_filter(
                kp1,
                desc1,
                feat1,
                policy,
                img1.shape[1],
                img1.shape[0]
            )

        if feat2 is not None:
            kp2, desc2 = apply_rl_filter(
                kp2,
                desc2,
                feat2,
                policy,
                img2.shape[1],
                img2.shape[0]
            )

    # --------------------------------------------------------
    # Matching
    # --------------------------------------------------------

    good_matches = get_good_matches(
        desc1,
        desc2,
        detector_name
    )

    # --------------------------------------------------------
    # RANSAC
    # --------------------------------------------------------

    inliers = get_inliers(
        kp1,
        kp2,
        good_matches
    )

    # --------------------------------------------------------
    # MMA
    # --------------------------------------------------------

    mma = compute_mma(
        kp1,
        kp2,
        good_matches,
        H_gt
    )

    return mma, inliers


# ============================================================
# CATEGORY EVALUATION
# ============================================================

def run_category_experiment(
    detector_name,
    policy,
    category
):

    sequences = get_sequences()

    category_sequences = []

    for seq in sequences:

        if category == "viewpoint" and seq.startswith("v_"):
            category_sequences.append(seq)

        elif category == "illumination" and seq.startswith("i_"):
            category_sequences.append(seq)

    total_mma = np.zeros(3)
    count = 0

    print(
        f"\n{detector_name} - "
        f"{category.upper()} - FULL"
    )

    for seq in category_sequences:

        for pair_number in range(2, 7):

            try:
                img1, img2, H_gt = read_pair(
                    seq,
                    pair_number
                )

                mma, inliers = evaluate_pair(
                    img1,
                    img2,
                    H_gt,
                    detector_name,
                    policy=None
                )

                total_mma += np.array(mma)
                count += 1

            except Exception as e:
                print(
                    f"Error: {seq} pair {pair_number}: {e}"
                )

    if count > 0:
        full_result = total_mma / count
    else:
        full_result = np.zeros(3)

    print(
        "Full MMA@1/3/5 = "
        f"{full_result[0]:.2f} / "
        f"{full_result[1]:.2f} / "
        f"{full_result[2]:.2f}"
    )

    # --------------------------------------------------------
    # RL
    # --------------------------------------------------------

    total_mma = np.zeros(3)
    count = 0

    print(
        f"\n{detector_name} - "
        f"{category.upper()} - RL"
    )

    for seq in category_sequences:

        for pair_number in range(2, 7):

            try:
                img1, img2, H_gt = read_pair(
                    seq,
                    pair_number
                )

                mma, inliers = evaluate_pair(
                    img1,
                    img2,
                    H_gt,
                    detector_name,
                    policy=policy
                )

                total_mma += np.array(mma)
                count += 1

            except Exception as e:
                print(
                    f"Error: {seq} pair {pair_number}: {e}"
                )

    if count > 0:
        rl_result = total_mma / count
    else:
        rl_result = np.zeros(3)

    print(
        "RL MMA@1/3/5 = "
        f"{rl_result[0]:.2f} / "
        f"{rl_result[1]:.2f} / "
        f"{rl_result[2]:.2f}"
    )

    return full_result, rl_result


# ============================================================
# TIMING EXPERIMENT
# ============================================================

def timing_experiment(
    detector_name,
    policy,
    repetitions=3
):

    sequences = get_sequences()

    full_times = []
    rl_times = []

    print(
        f"\nTiming {detector_name}..."
    )

    # Warm-up
    for seq in sequences[:2]:

        try:
            img1, img2, H_gt = read_pair(
                seq,
                2
            )

            evaluate_pair(
                img1,
                img2,
                H_gt,
                detector_name,
                policy=None
            )

            evaluate_pair(
                img1,
                img2,
                H_gt,
                detector_name,
                policy=policy
            )

        except Exception:
            pass

    # --------------------------------------------------------
    # Measure several repetitions
    # --------------------------------------------------------

    for _ in range(repetitions):

        start = time.perf_counter()

        count = 0

        for seq in sequences:

            for pair_number in range(2, 7):

                try:
                    img1, img2, H_gt = read_pair(
                        seq,
                        pair_number
                    )

                    evaluate_pair(
                        img1,
                        img2,
                        H_gt,
                        detector_name,
                        policy=None
                    )

                    count += 1

                except Exception:
                    pass

        elapsed = time.perf_counter() - start

        if count > 0:
            full_times.append(
                elapsed * 1000 / count
            )

    for _ in range(repetitions):

        start = time.perf_counter()

        count = 0

        for seq in sequences:

            for pair_number in range(2, 7):

                try:
                    img1, img2, H_gt = read_pair(
                        seq,
                        pair_number
                    )

                    evaluate_pair(
                        img1,
                        img2,
                        H_gt,
                        detector_name,
                        policy=policy
                    )

                    count += 1

                except Exception:
                    pass

        elapsed = time.perf_counter() - start

        if count > 0:
            rl_times.append(
                elapsed * 1000 / count
            )

    full_avg = np.mean(full_times)
    rl_avg = np.mean(rl_times)

    print(
        f"{detector_name} Full: "
        f"{full_avg:.2f} ms/image pair"
    )

    print(
        f"{detector_name} RL: "
        f"{rl_avg:.2f} ms/image pair"
    )

    return full_avg, rl_avg


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ORB + BRISK EXTRA HPATCHES EVALUATION")
    print("=" * 70)

    print(
        f"Found {len(get_sequences())} HPatches sequences."
    )

    # --------------------------------------------------------
    # ORB
    # --------------------------------------------------------

    print("\n\n================ ORB ================")

    orb_policy = load_policy(ORB_MODEL)

    orb_vp = run_category_experiment(
        "ORB",
        orb_policy,
        "viewpoint"
    )

    orb_illum = run_category_experiment(
        "ORB",
        orb_policy,
        "illumination"
    )

    orb_time = timing_experiment(
        "ORB",
        orb_policy
    )

    # --------------------------------------------------------
    # BRISK
    # --------------------------------------------------------

    print("\n\n================ BRISK ================")

    brisk_policy = load_policy(BRISK_MODEL)

    brisk_vp = run_category_experiment(
        "BRISK",
        brisk_policy,
        "viewpoint"
    )

    brisk_illum = run_category_experiment(
        "BRISK",
        brisk_policy,
        "illumination"
    )

    brisk_time = timing_experiment(
        "BRISK",
        brisk_policy
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n\n")
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print("\nORB")
    print(
        "Viewpoint Full: "
        f"{orb_vp[0][0]:.2f}, "
        f"{orb_vp[0][1]:.2f}, "
        f"{orb_vp[0][2]:.2f}"
    )
    print(
        "Viewpoint RL:   "
        f"{orb_vp[1][0]:.2f}, "
        f"{orb_vp[1][1]:.2f}, "
        f"{orb_vp[1][2]:.2f}"
    )

    print(
        "Illumination Full: "
        f"{orb_illum[0][0]:.2f}, "
        f"{orb_illum[0][1]:.2f}, "
        f"{orb_illum[0][2]:.2f}"
    )
    print(
        "Illumination RL:   "
        f"{orb_illum[1][0]:.2f}, "
        f"{orb_illum[1][1]:.2f}, "
        f"{orb_illum[1][2]:.2f}"
    )

    print(
        f"Timing Full: {orb_time[0]:.2f} ms"
    )
    print(
        f"Timing RL:   {orb_time[1]:.2f} ms"
    )

    print("\nBRISK")
    print(
        "Viewpoint Full: "
        f"{brisk_vp[0][0]:.2f}, "
        f"{brisk_vp[0][1]:.2f}, "
        f"{brisk_vp[0][2]:.2f}"
    )
    print(
        "Viewpoint RL:   "
        f"{brisk_vp[1][0]:.2f}, "
        f"{brisk_vp[1][1]:.2f}, "
        f"{brisk_vp[1][2]:.2f}"
    )

    print(
        "Illumination Full: "
        f"{brisk_illum[0][0]:.2f}, "
        f"{brisk_illum[0][1]:.2f}, "
        f"{brisk_illum[0][2]:.2f}"
    )
    print(
        "Illumination RL:   "
        f"{brisk_illum[1][0]:.2f}, "
        f"{brisk_illum[1][1]:.2f}, "
        f"{brisk_illum[1][2]:.2f}"
    )

    print(
        f"Timing Full: {brisk_time[0]:.2f} ms"
    )
    print(
        f"Timing RL:   {brisk_time[1]:.2f} ms"
    )


if __name__ == "__main__":
    main()
