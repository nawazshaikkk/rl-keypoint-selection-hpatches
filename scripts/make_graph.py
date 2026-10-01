import csv
import os
from collections import defaultdict
import matplotlib.pyplot as plt

INPUT = "results/all_results.csv"
OUTPUT = "results/hpatches_benchmark.png"

# Store values for each descriptor
data = defaultdict(lambda: {
    "Precision": [],
    "MMA@1px": [],
    "MMA@3px": [],
    "MeanError": [],
    "TimeSeconds": []
})

# Read CSV
with open(INPUT, "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        method = row["Method"]

        data[method]["Precision"].append(float(row["Precision"]))
        data[method]["MMA@1px"].append(float(row["MMA@1px"]))
        data[method]["MMA@3px"].append(float(row["MMA@3px"]))
        data[method]["MeanError"].append(float(row["MeanError"]))
        data[method]["TimeSeconds"].append(float(row["TimeSeconds"]))


# Methods
methods = ["SIFT", "ORB", "BRISK", "LATCH"]


# Calculate averages
def average(values):
    return sum(values) / len(values)


precision = [average(data[m]["Precision"]) for m in methods]
mma1 = [average(data[m]["MMA@1px"]) for m in methods]
mma3 = [average(data[m]["MMA@3px"]) for m in methods]

# Convert seconds to milliseconds
time_ms = [
    average(data[m]["TimeSeconds"]) * 1000
    for m in methods
]

mean_error = [
    average(data[m]["MeanError"])
    for m in methods
]


# ------------------------------------------------
# Create figure
# ------------------------------------------------

fig, axes = plt.subplots(1, 5, figsize=(18, 4.8))

fig.suptitle(
    "HPatches Descriptor Benchmark",
    fontsize=16,
    fontweight="bold"
)


# 1. MMA @ 1px
axes[0].bar(methods, mma1)

axes[0].set_title("MMA @ 1px ↑")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0, 1)

for i, value in enumerate(mma1):
    axes[0].text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
        fontsize=9
    )


# 2. MMA @ 3px
axes[1].bar(methods, mma3)

axes[1].set_title("MMA @ 3px ↑")
axes[1].set_ylim(0, 1)

for i, value in enumerate(mma3):
    axes[1].text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
        fontsize=9
    )


# 3. Precision
axes[2].bar(methods, precision)

axes[2].set_title("Precision ↑")
axes[2].set_ylim(0, 1)

for i, value in enumerate(precision):
    axes[2].text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
        fontsize=9
    )


# 4. Mean Error
axes[3].bar(methods, mean_error)

axes[3].set_title("Mean Error ↓")
axes[3].set_ylabel("Pixels")

for i, value in enumerate(mean_error):
    axes[3].text(
        i,
        value + 1,
        f"{value:.1f}",
        ha="center",
        fontsize=9
    )


# 5. Speed
axes[4].bar(methods, time_ms)

axes[4].set_title("Execution Time ↓")
axes[4].set_ylabel("Milliseconds")

for i, value in enumerate(time_ms):
    axes[4].text(
        i,
        value + max(time_ms) * 0.02,
        f"{value:.1f}",
        ha="center",
        fontsize=9
    )


# Common formatting
for ax in axes:
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=25)


# Subtitle / conclusion
fig.text(
    0.5,
    0.01,
    "Evaluation across 116 HPatches sequences (2320 descriptor evaluations)",
    ha="center",
    fontsize=11
)

plt.tight_layout(rect=[0, 0.06, 1, 0.93])


# Save PNG
os.makedirs("results", exist_ok=True)

plt.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("=" * 60)
print("BENCHMARK IMAGE GENERATED")
print("=" * 60)
print("Saved to:", OUTPUT)
