import cv2
import numpy as np


for image_number in range(2, 7):

    img1 = cv2.imread(
        "hpatches/i_ajuntament/1.ppm",
        cv2.IMREAD_GRAYSCALE
    )

    img2 = cv2.imread(
        "hpatches/i_ajuntament/" + str(image_number) + ".ppm",
        cv2.IMREAD_GRAYSCALE
    )

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    raw_matches = bf.knnMatch(
        des1,
        des2,
        k=2
    )

    good_matches = []

    for pair in raw_matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    inliers = 0

    if len(good_matches) >= 4:

        src_pts = np.float32([
            kp1[m.queryIdx].pt
            for m in good_matches
        ]).reshape(-1, 1, 2)

        dst_pts = np.float32([
            kp2[m.trainIdx].pt
            for m in good_matches
        ]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            5.0
        )

        if mask is not None:
            inliers = int(np.sum(mask))

    print(
        "1 ->",
        image_number,
        "| Keypoints:",
        len(kp1),
        "+",
        len(kp2),
        "| Good:",
        len(good_matches),
        "| RANSAC:",
        inliers
    )
