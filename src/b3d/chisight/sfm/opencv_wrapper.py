"""
Wrapper for OpenCV functions for SIFT and ORB feature detection and matching.
For convenience mostly.
"""

import cv2
import numpy as np

from b3d.pose import Pose


def detect_orb(img, n=500, **kwargs):
    """
    Wrapper for ORB detector `cv2.SIFT.detectAndCompute`.
    """
    img = np.array(img)
    orb = cv2.ORB_create(nfeatures=n)
    kps, des = orb.detectAndCompute(img, None, **kwargs)
    uvs = np.array([kp.pt for kp in kps])
    return uvs, des


def detect_sift(img):
    """
    Wrapper for SIFT detector `cv2.SIFT.detectAndCompute`.
    """
    img = np.array(img)
    sift = cv2.SIFT_create()
    kps, des = sift.detectAndCompute(img, None)
    uvs = np.array([kp.pt for kp in kps])
    return uvs, des


def match_bf(des0, des1, thresh=0.75):
    """
    Wrapper for brute force mathcer `cv2.BFMatcher.knnMatch`.
    """
    # Matched indices
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des0, des1, k=2)
    inds = []
    for m, n in matches:
        if m.distance < thresh * n.distance:
            i0 = m.queryIdx
            i1 = m.trainIdx
            inds.append((i0, i1))
    inds = np.array(inds)
    return inds


def detect_and_match_sift(img0, img1, thresh=0.75):
    """
    Returns:
        - (uvs0, des0): SIFT keypoint positions and descriptors of img0
        - (uvs1, des1): SIFT keypoint positions and descriptors of img1
        - inds: np.array of shape (M,2) of matched indices
    """
    img0 = np.array(img0)
    img1 = np.array(img1)
    sift = cv2.SIFT_create()
    kp0, des0 = sift.detectAndCompute(img0, None)
    kp1, des1 = sift.detectAndCompute(img1, None)
    uvs0 = np.array([kp.pt for kp in kp0])
    uvs1 = np.array([kp.pt for kp in kp1])

    # Matched indices
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des0, des1, k=2)
    inds = []
    for m, n in matches:
        if m.distance < thresh * n.distance:
            i0 = m.queryIdx
            i1 = m.trainIdx
            inds.append((i0, i1))
    inds = np.array(inds)

    return (uvs0, des0, inds[:, 0]), (uvs1, des1, inds[:, 1])


def match_sift(des0, des1, thresh=0.75):
    # Matched indices
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des0, des1, k=2)
    inds = []
    for m, n in matches:
        if m.distance < thresh * n.distance:
            i0 = m.queryIdx
            i1 = m.trainIdx
            inds.append((i0, i1))
    inds = np.array(inds)
    return inds


def matched_only_sift(img0, img1, thresh=0.75):
    (uvs0, _, inds0), (uvs1, _, inds1) = detect_and_match_sift(
        img0, img1, thresh=thresh
    )
    pts0 = uvs0[inds0]
    pts1 = uvs1[inds1]
    return pts0, pts1


def recover_pose(E, uvs0, uvs1, cam_K):
    _, R, t, _ = cv2.recoverPose(
        np.array(E).astype(np.float64),
        points1=np.array(uvs0),
        points2=np.array(uvs1),
        cameraMatrix=np.array(cam_K),
    )
    p = Pose.from_pos_matrix(t[:, 0], R).inv()
    return p


def find_essential(pts0, pts1, cam_K, prob=0.999999, thresh=1.0):
    pts0 = np.array(pts0)
    pts1 = np.array(pts1)
    cam_K = np.array(cam_K)

    E, mask = cv2.findEssentialMat(
        pts0, pts1, cam_K, cv2.RANSAC, prop=prob, threshold=thresh
    )
    inlier = mask[:, 0] == 1
    return E, inlier


def infer_pose(pts0, pts1, cam_K, prob=0.999999, threshold=1.0, max_iters=1000):
    pts0 = np.array(pts0)
    pts1 = np.array(pts1)
    cam_K = np.array(cam_K)

    E, _ = cv2.findEssentialMat(
        pts0,
        pts1,
        cam_K,
        cv2.RANSAC,
        prob=prob,
        threshold=threshold,
        maxIters=max_iters,
    )

    _, R, t, _ = cv2.recoverPose(E, points1=pts0, points2=pts1, cameraMatrix=cam_K)
    p = Pose.from_pos_matrix(t[:, 0], R).inv()
    return p


def infer_essential(pts0, pts1, cam_K, prob=0.999999, threshold=1.0, max_iters=1000):
    pts0 = np.array(pts0)
    pts1 = np.array(pts1)
    cam_K = np.array(cam_K)

    E, mask = cv2.findEssentialMat(
        pts0,
        pts1,
        cam_K,
        cv2.RANSAC,
        prob=prob,
        threshold=threshold,
        maxIters=max_iters,
    )
    inlier = mask[:, 0] == 1
    return E, inlier


def infer_pose_and_inlier(
    pts0, pts1, cam_K, prob=0.999999, threshold=1.0, max_iters=1000
):
    pts0 = np.array(pts0)
    pts1 = np.array(pts1)
    cam_K = np.array(cam_K)

    E, mask = cv2.findEssentialMat(
        pts0,
        pts1,
        cam_K,
        cv2.RANSAC,
        prob=prob,
        threshold=threshold,
        maxIters=max_iters,
    )
    inlier = mask[:, 0] == 1

    _, R, t, _ = cv2.recoverPose(E, points1=pts0, points2=pts1, cameraMatrix=cam_K)
    p = Pose.from_pos_matrix(t[:, 0], R).inv()
    return p, inlier
