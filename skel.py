#!/home/panyu/Devs/miniconda3/bin/python

from trans import Trans

import math
import numpy as np
import open3d as o3d
import yaml


class SkeletonPart:
    def __init__(self):
        self.confidence_level = 0
        self.trans = Trans()
    
    def __str__(self) -> str:
        return (
            f"confidence_level: {self.confidence_level}\n"
            f"trans: {self.trans}\n"
        )


class Skeleton:
    def __init__(self):
        self.index = 0
        self.skeleton_parts = []  # list of SkeletonPart
    
    def __str__(self) -> str:
        expr = f"index: {self.index}\n"
        for s in self.skeleton_parts:
            expr = expr + f"{s}"
        return expr


class SkeletonFrame:

    # Specifies the lines connecting the skeleton joints in terms of joint indices,
    # according to https://docs.microsoft.com/en-us/azure/kinect-dk/body-joints
    JOINT_LINES = (
        (0,1), (1,2), (2,3),
        (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,10),
        (3,11), (11,12), (12,13), (13,14), (14,15), (15,16), (16,17),
        (0,18), (18,19), (19,20), (20,21),
        (0,22), (22,23), (23,24), (24,25),
        (3,26), (26,27), (27,28), (28,29), (27,30), (27,31)
    )

    def __init__(self):
        self.num_skeletons = 0
        self.skeleton_list = []   # list of Skeleton

    
    def __str__(self) -> str:
        expr = f"======== SkeletonFrame ========\n"
        for l in self.skeleton_list:
            expr = expr + f"{l}\n"
        expr = expr + f"======== Enfof SkeletonFrame ========\n"
        return expr


    def from_yaml(self, skel_file: str, kinect_pose: Trans = None) -> None:
        self.num_skeletons = 0
        self.skeleton_list = []

        with open(skel_file, 'r') as stream:
            try:
                skelentons = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print(e)
        
        if kinect_pose is not None:
            map2kinect4x4 = kinect_pose.get_tf_mat4x4()
        else:
            map2kinect4x4 = np.eye(4, 4)
    
        for s in skelentons:
            skel = Skeleton()
            skel.index = s["idx"]
            for p in s["skeleton"]:
                skel_part = SkeletonPart()
                skel_part.confidence_level = p["confidence_level"]
                skel_part.trans.rot_quat = p["orientation_qwxyz"]
                skel_part.trans.trans_vec = p["position"]
                skel_part.trans.from_mat4x4(np.matmul(map2kinect4x4, skel_part.trans.get_tf_mat4x4()))
                skel.skeleton_parts.append(skel_part)
            self.skeleton_list.append(skel)
        self.num_skeletons = len(self.skeleton_list)

    
    def get_interpolated_skel_pts(self, skel_idx: int, gap: float = 0.05) -> np.ndarray:
        if self.skeleton_list is None or skel_idx > len(self.skeleton_list) - 1:
            return None
        
        whole_pts_list = []
        skel: Skeleton = self.skeleton_list[skel_idx]
        for line in self.JOINT_LINES:
            start_pt = skel.skeleton_parts[line[0]].trans.trans_vec
            end_pt = skel.skeleton_parts[line[1]].trans.trans_vec
            dx = end_pt[0] - start_pt[0]
            dy = end_pt[1] - start_pt[1]
            dz = end_pt[2] - start_pt[2]
            dist = math.hypot(dx, dy, dz)
            num_segs = math.ceil(dist / gap)
            line_pts = [start_pt]
            if num_segs > 1:
                for i in range(num_segs - 1):
                    line_pts.append(
                        [
                            start_pt[0] + (i+1) * dx / num_segs,
                            start_pt[1] + (i+1) * dy / num_segs,
                            start_pt[2] + (i+1) * dz / num_segs
                        ]
                    )
            line_pts.append(end_pt)
            whole_pts_list.extend(line_pts)
        
        return_pts = np.array(whole_pts_list)
        return return_pts


    def get_tight_skel_bounding_box(self, skel_idx: int) -> o3d.geometry.AxisAlignedBoundingBox:
        if self.skeleton_list is None or skel_idx > len(self.skeleton_list) - 1:
            return None
        
        skel_pts_list = []
        skel: Skeleton = self.skeleton_list[skel_idx]
        for skel_part in skel.skeleton_parts:
            skel_pts_list.append(skel_part.trans.trans_vec)
        
        skel_pcd = o3d.geometry.PointCloud()
        skel_pcd.points = o3d.utility.Vector3dVector(np.array(skel_pts_list))
        return skel_pcd.get_axis_aligned_bounding_box()
