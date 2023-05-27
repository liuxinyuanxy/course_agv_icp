#!/usr/bin/env python

import rospy
import numpy as np
import math
import scipy as sp


# structure of the nearest neighbor
class NeighBor:
    def __init__(self):
        self.distances = []
        self.src_indices = []
        self.tar_indices = []


class ICP:
    def __init__(self):
        # max iterations
        self.max_iter = rospy.get_param('/icp/max_iter', 10)
        # distance threshold for filter the matching points
        self.dis_th = rospy.get_param('/icp/dis_th', 3)
        # tolerance to stop icp
        self.tolerance = rospy.get_param('/icp/tolerance', 0)
        # min match
        self.min_match = rospy.get_param('/icp/min_match', 2)

    # ICP process function
    # Waiting for Implementation 
    # return: T = (R, t), where T is 2*3, R is 2*2 and t is 2*1
    def process(self, tar_pc: np.ndarray, src_pc: np.ndarray):
        # clean the nan
        tar_pc = tar_pc[~np.isnan(tar_pc).any(axis=1)]
        src_pc = src_pc[~np.isnan(src_pc).any(axis=1)]
        T = np.array([[1, 0, 0], [0, 1, 0]])
        # do the iteration
        for i in range(self.max_iter):
            # find the nearest points
            neigh = self.findNearest(src_pc, tar_pc)
            # check the number of the nearest points
            if len(neigh.distances) < self.min_match:
                print("No enough points")
                return None
            # get the transform
            Temp = self.getTransform(src_pc[neigh.src_indices], tar_pc[neigh.tar_indices])
            # update the src_pc
            for j in range(len(src_pc)):
                src_pc[j] = np.dot(Temp, np.append(src_pc[j], 1))[:2]
            # update the transform
            T = np.dot(Temp, T)
            # check the tolerance
            if np.linalg.norm(Temp - np.eye(2)) < self.tolerance:
                break
        return T

    def calcDist(self, a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.hypot(dx, dy)

    # find the nearest points & filter
    # use kd tree to find the nearest neighbor
    # return: neighbors of src and tar
    def findNearest(self, src, tar):
        kd_tree = sp.spatial.KDTree(tar)
        neigh = NeighBor()
        for i in range(len(src)):
            dist, indices = kd_tree.query(src[i])
            if dist < self.dis_th:
                neigh.distances.append(dist)
                neigh.src_indices.append(i)
                neigh.tar_indices.append(indices)
        return neigh

    # Waiting for Implementation
    # return: T = (R, t), where T is 2*3, R is 2*2 and t is 2*1
    def getTransform(self, src, tar):
        # get src vector
        src_mean = np.mean(src, axis=0)
        src_vec = src - src_mean
        # get tar vector
        tar_mean = np.mean(tar, axis=0)
        tar_vec = tar - tar_mean
        # get covariance matrix
        H = np.matmul(src_vec.T, tar_vec)
        # SVD
        U, S, V = np.linalg.svd(H)
        # get rotation matrix
        R = np.matmul(V.T, U.T)
        # get translation matrix
        t = tar_mean - np.matmul(R, src_mean)
        # get transform matrix
        T = np.vstack((R, t))
        return T
