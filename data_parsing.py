#!/home/panyu/Devs/miniconda3/bin/python

from trans import Trans
from skel import *

import glob
import os
import open3d as o3d
import pandas as pd
from typing import List, Tuple


class PcdDataFiles:
    def __init__(self):
        self.ts: str = ""
        self.kinect_pose = Trans()
        self.pcd_file = ""
        self.body_in_map_pcd_file = ""
        self.skel_file = ""
        self.res_file = ""
    
    def __str__(self) -> str:
        return (
            f"======== PcdDataFiles frame ========\n"
            f"ts: {self.ts}\n"
            f"kinect_pose: {self.kinect_pose}\n"
            f"pcd_file: {self.pcd_file}\n"
            f"body_in_map_pcd_file: {self.body_in_map_pcd_file}\n"
            f"skel_file: {self.skel_file}\n"
            f"res_file: {self.res_file}\n"
            f"======== Endof PcdDataFiles ========\n"
        )


class PcdDataFrame:
    def __init__(self):
        self.ts: int = 0
        self.kinect_pose = Trans()
        self.pcd = o3d.geometry.PointCloud()
        self.body_pcd = o3d.geometry.PointCloud()
        self.skel_frame = SkeletonFrame()
        self.res_pcd = o3d.geometry.PointCloud()
    
    def from_PcdDataFiles(self, pcd_data_files_frame: PcdDataFiles) -> None:
        f_frame = pcd_data_files_frame
        self.ts = int(f_frame.ts)
        self.kinect_pose = f_frame.kinect_pose

        self.pcd = o3d.io.read_point_cloud(f_frame.pcd_file)

        if (f_frame.body_in_map_pcd_file is None or f_frame.body_in_map_pcd_file == ""):
            self.body_pcd = None
        else:
            self.body_pcd = o3d.io.read_point_cloud(pcd_data_files_frame.body_in_map_pcd_file)

        if (f_frame.skel_file is None or f_frame.skel_file == ""):
            self.skel_frame = None
        else:
            self.skel_frame.from_yaml(pcd_data_files_frame.skel_file, self.kinect_pose)
        
        if f_frame.res_file:
            self.res_pcd = o3d.io.read_point_cloud(f_frame.res_file)


def get_pcd_data_files_list(data_folder: str = "data") -> List[PcdDataFiles]:
    scan_in_map_pcd_dir = os.path.join(data_folder, "pcd")
    scan_in_map_pcd_files = glob.glob(scan_in_map_pcd_dir + "/*.pcd")
    scan_in_map_pcd_files = list(filter(lambda f: os.path.isfile(f), scan_in_map_pcd_files))
    scan_in_map_pcd_files = [os.path.split(p)[1] for p in scan_in_map_pcd_files]
    scan_in_map_pcd_files.sort()
    print(f"number of scan_in_map_pcd_files: {len(scan_in_map_pcd_files)}")
    
    ts_list = []
    for l in scan_in_map_pcd_files:
        ts_list.append(str(int(os.path.splitext(l)[0])))
    print(f"number of ts_list_with_body_detections: {len(ts_list)}")

    body_in_map_pcd_dir = os.path.join(data_folder, "body")
    body_in_map_pcd_files = glob.glob(body_in_map_pcd_dir + "/*.pcd")
    body_in_map_pcd_files = list(filter(lambda f: os.path.isfile(f), body_in_map_pcd_files))
    body_in_map_pcd_files = [os.path.split(p)[1] for p in body_in_map_pcd_files]
    body_in_map_pcd_files.sort()
    print(f"number of body_in_map_pcd_files: {len(body_in_map_pcd_files)}")

    skel_dir = os.path.join(data_folder, "skeleton")
    skel_files = glob.glob(skel_dir + "/*.yaml")
    skel_files = list(filter(lambda f: os.path.isfile(f), skel_files))
    skel_files = [os.path.split(p)[1] for p in skel_files]
    skel_files.sort()
    print(f"number of skel_files: {len(skel_files)}")

    poses_csv_file = os.path.join(data_folder, "poses.csv")
    if not os.path.isfile(poses_csv_file):
        raise Exception(f"pose file doesn't exist")
    poses_csv = pd.read_csv(poses_csv_file, delimiter = ',')
    ts_matched_dataframe = poses_csv[poses_csv['ts'].isin(ts_list)]

    res_dir = os.path.join(data_folder, "result")

    # Get the pcd_data_files list
    pcd_data_files_list = []
    for _, row in ts_matched_dataframe.iterrows():
        ts = f"{int(row.ts):010d}"
        cur_info = PcdDataFiles()
        cur_info.ts = ts
        cur_info.kinect_pose.trans_vec = [row.x, row.y, row.z]
        cur_info.kinect_pose.rot_quat = [row.qw, row.qx, row.qy, row.qz]
        cur_info.kinect_pose.rpy = [row.rx, row.ry, row.rz]

        cur_info.pcd_file = os.path.join(scan_in_map_pcd_dir, ts+".pcd")
        assert os.path.isfile(cur_info.pcd_file), f"Not able to find {cur_info.pcd_file}"

        cur_info.body_in_map_pcd_file = os.path.join(body_in_map_pcd_dir, ts+".pcd")
        if not os.path.isfile(cur_info.body_in_map_pcd_file):
            cur_info.body_in_map_pcd_file = ""

        cur_info.skel_file = os.path.join(skel_dir, ts+".yaml")
        if not os.path.isfile(cur_info.skel_file):
            cur_info.skel_file = ""
        
        cur_info.res_file = os.path.join(res_dir, ts+".pcd")
        if not os.path.isfile(cur_info.res_file):
            cur_info.res_file = ""
        
        pcd_data_files_list.append(cur_info)
    
    print(f"number of pcd_data_files: {len(pcd_data_files_list)}")

    return pcd_data_files_list
