import matplotlib.pyplot as plt


# BRISK baseline
brisk = [
    45.52,
    77.63,
    84.11
]

# BRISK + RL
brisk_rl = [
    40.69,
    71.96,
    79.74
]

thresholds = [
    "1 px",
    "3 px",
    "5 px"
]

x = range(len(thresholds))

width = 0.35


plt.figure(figsize=(9, 5))

plt.bar(
    [i - width / 2 for i in x],
    brisk,
    width,
    label="BRISK"
)

plt.bar(
    [i + width / 2 for i in x],
    brisk_rl,
    width,
    label="BRISK + RL"
)

plt.xticks(
    list(x),
    thresholds
)

plt.ylabel(
    "Mean Matching Accuracy (%)"
)

plt.xlabel(
    "Pixel error threshold"
)

plt.title(
    "BRISK vs BRISK + RL on Complete HPatches Dataset"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/brisk_before_after_accuracy.png",
    dpi=300
)

plt.close()

print("Graph saved:")
print("results/brisk_before_after_accuracy.png")
