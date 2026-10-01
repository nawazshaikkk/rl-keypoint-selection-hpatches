# Reinforcement Learning Based Keypoint Selection for Efficient Image Matching

This project explores reinforcement-learning-based keypoint selection for
efficient image matching using ORB, BRISK, and SIFT.

## Project Idea

Classical feature detectors can generate many keypoints, but not every
keypoint is useful for matching.

We add a lightweight reinforcement learning layer after feature detection.
The RL policy decides whether each keypoint should be:

- KEEP
- REJECT

Selected keypoints are then passed to descriptor matching and RANSAC
geometric verification.

## Pipeline

Detector → Keypoint Detection → RL Selection → Descriptor Matching → RANSAC → MMA

## Methods

- ORB
- BRISK
- SIFT

## Dataset

HPatches

- 116 sequences
- 580 image pairs
- Viewpoint and illumination variations

## RL Policy

8-dimensional state:

1. Normalized X position
2. Normalized Y position
3. Keypoint scale
4. Keypoint orientation
5. Detector response
6. Best descriptor distance
7. Second-best descriptor distance
8. Distance ratio

Network:

8 → 32 → 16 → 2

Actions:

- KEEP
- REJECT

## Training

Algorithm: REINFORCE  
Optimizer: Adam  
Episodes: 10,000

Learning rates:

- ORB: 1 × 10^-4
- BRISK: 1 × 10^-4
- SIFT: 5 × 10^-5

## Results

| Detector | Full Keypoints | After RL | Reduction |
|----------|---------------:|---------:|----------:|
| ORB | 998.47 | 636.43 | 36.26% |
| BRISK | 14,879.34 | 1,387.71 | 90.67% |
| SIFT | 9,668.77 | 5,254.01 | 45.66% |

## Evaluation Metrics

- Keypoint reduction
- RANSAC inliers
- MMA@1
- MMA@3
- MMA@5
- Wall-clock time

## Project Structure

```text
scripts/   - Python source code
results/   - graphs and evaluation results
models/    - trained RL models
