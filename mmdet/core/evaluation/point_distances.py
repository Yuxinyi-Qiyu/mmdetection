# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

def point_distances(points1,
                  points2,
                  mode='l2'):
    """Calculate the dis between each bbox of points1 and points2.

    Args:
        points1 (ndarray): Shape (n, 2)
        points2 (ndarray): Shape (k, 2)
        mode (str): l2 (l2 distance)

    Returns:
        distances (ndarray): Shape (n, k)
    """

    assert mode in ['l1', 'l2']
    points1 = points1.astype(np.float32)
    points2 = points2.astype(np.float32)
    rows = points1.shape[0]
    cols = points2.shape[0]
    dis = np.ones((rows, cols), dtype=np.float32)*1e6
    if rows * cols == 0:
        return dis
    exchange = False
    if points1.shape[0] > points2.shape[0]:
        points1, points2 = points2, points1
        dis = np.ones((cols, rows), dtype=np.float32)*1e6
        exchange = True
    for i in range(points1.shape[0]):
        dis_x = np.abs(points1[i, 0] - points2[:, 0])
        dis_y = np.abs(points1[i, 1] - points2[:, 1])
        dis[i, :] = dis_x + dis_y if mode == 'l1' else np.sqrt(dis_x**2+dis_y**2)
    if exchange:
        dis = dis.T
    return dis

def point_in_bbox(points,
                  bboxes,
                  mode='l2',
                  inf=1e6):
    """Calculate the dis between each bbox of points1 and points2.

    Args:
        points (ndarray): Shape (n, 2)
        bbox (ndarray): Shape (k, 2)
        mode (str): l2 (l2 distance)

    Returns:
        distance (ndarray): Shape (n, k)
    """

    assert mode in ['l1', 'l2']
    points = points.astype(np.float32)
    bboxes = bboxes.astype(np.float32)
    rows = points.shape[0]
    cols = bboxes.shape[0]
    dis = np.ones((rows, cols), dtype=np.float32)*inf
    if rows * cols == 0:
        return dis
    for i in range(points.shape[0]):
        dis[i, :] = np.where(bboxes[:, 0]<=points[i, 0], 0, inf)
        dis[i, :] = np.where(bboxes[:, 2]>=points[i, 0], dis[i, :], inf)
        dis[i, :] = np.where(bboxes[:, 1]<=points[i, 1], dis[i, :], inf)
        dis[i, :] = np.where(bboxes[:, 3]>=points[i, 1], dis[i, :], inf)
    return dis