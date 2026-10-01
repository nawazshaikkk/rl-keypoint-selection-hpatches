import matplotlib.pyplot as plt
import os


# ==================================================
# RESULTS FROM OUR FINAL EXPERIMENT
# ==================================================

orb_keypoints = 999.98
rl_keypoints = 425.02

orb_inliers = 86.25
rl_inliers = 70.97

orb_mma1 = 37.57
orb_mma3 = 67.85
orb_mma5 = 75.08

rl_mma1 = 38.11
rl_mma3 = 65.94
rl_mma5 = 72.93


# ==================================================
# CREATE RESULTS FOLDER
# ==================================================

os.makedirs(
    "results",
    exist_ok=True
)


# ==================================================
# GRAPH 1
# KEYPOINT COMPARISON
# ==================================================

models = [
    "ORB",
    "ORB + RL"
]

keypoints = [
    orb_keypoints,
    rl_keypoints
]


plt.figure(
    figsize=(8, 5)
)

plt.bar(
    models,
    keypoints
)

plt.ylabel(
    "Average Number of Keypoints"
)

plt.title(
    "Average Keypoints: ORB vs ORB + RL"
)

plt.tight_layout()

plt.savefig(
    "results/keypoints_comparison.png",
    dpi=300
)

plt.close()


# ==================================================
# GRAPH 2
# RANSAC INLIER COMPARISON
# ==================================================

inliers = [
    orb_inliers,
    rl_inliers
]


plt.figure(
    figsize=(8, 5)
)

plt.bar(
    models,
    inliers
)

plt.ylabel(
    "Average RANSAC Inliers"
)

plt.title(
    "Average RANSAC Inliers: ORB vs ORB + RL"
)

plt.tight_layout()

plt.savefig(
    "results/ransac_comparison.png",
    dpi=300
)

plt.close()


# ==================================================
# GRAPH 3
# MMA COMPARISON
# ==================================================

thresholds = [
    "1 px",
    "3 px",
    "5 px"
]


orb_mma = [
    orb_mma1,
    orb_mma3,
    orb_mma5
]


rl_mma = [
    rl_mma1,
    rl_mma3,
    rl_mma5
]


x = range(
    len(thresholds)
)


width = 0.35


plt.figure(
    figsize=(9, 5)
)


plt.bar(
    [i - width / 2 for i in x],
    orb_mma,
    width,
    label="ORB"
)


plt.bar(
    [i + width / 2 for i in x],
    rl_mma,
    width,
    label="ORB + RL"
)


plt.xticks(
    list(x),
    thresholds
)


plt.ylabel(
    "Mean Matching Accuracy (%)"
)

plt.xlabel(
    "Pixel Error Threshold"
)

plt.title(
    "Mean Matching Accuracy: ORB vs ORB + RL"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/mma_comparison.png",
    dpi=300
)

plt.close()


# ==================================================
# DONE
# ==================================================

print("=" * 50)
print("GRAPHS CREATED")
print("=" * 50)

print(
    "Saved: results/keypoints_comparison.png"
)

print(
    "Saved: results/ransac_comparison.png"
)

print(
    "Saved: results/mma_comparison.png"
)
