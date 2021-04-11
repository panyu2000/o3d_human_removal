#!/usr/bin/env python

from numpy.lib.shape_base import tile
from data_parsing import *
import tracking
from skel import *

import argparse
import math
import matplotlib.pyplot as plt 
import numpy as np
import open3d as o3d
import os
import shutil
import sys
from typing import List, Tuple

# Types
o3d_pcd = o3d.geometry.PointCloud
o3d_bb = o3d.geometry.AxisAlignedBoundingBox

# Global variables
glb_skel_bb_list = []
glb_track_man = tracking.TrackManager()
glb_tracked_top_bb_list = []
glb_tracked_bot_bb_list = []
    
UPRIGHT_TOLERANCE_DEG = 20  # Normals different within +- this range are similar to ground
Z_NORMAL_THOLD = math.cos(math.radians(UPRIGHT_TOLERANCE_DEG))


def get_sub_pcd_from_bounding_box(pcd: o3d.geometry.PointCloud,
                                  bb: o3d.geometry.AxisAlignedBoundingBox,
                                  color: list = [0., 0., 0.]) -> o3d.geometry.PointCloud :
    """ Crop out a sub pointcloud from the input pointcloud within the bounding box.
    """
    new_pcd = pcd.crop(bb)
    new_pcd.paint_uniform_color(color)
    return new_pcd


def exclude_pts_in_bounding_boxes(pcd: o3d.geometry.PointCloud,
                                  bb_list: list) -> o3d.geometry.PointCloud:
    """ Return a new PointCloud with points in the bb_list removed.
    """
    selected_indices = set()
    for bb in bb_list:
        # Slightly scale up bb, otherwise some boundry points won't be included (strange)
        enlarged_bb = bb.scale(1.01, bb.get_center())
        selected_indices = selected_indices.union(set(enlarged_bb.get_point_indices_within_bounding_box(pcd.points)))
    
    return pcd.select_by_index(list(selected_indices), invert=True)


def recover_full_scene_pcd(data_frame: PcdDataFrame) -> o3d_pcd:
    """ Merge the scene pointcloud (maybe with body pointcloud removed) with
        the body pointcloud, and return the merged full scene pointcloud.
    """
    full_scene_pcd = o3d_pcd(data_frame.pcd) # make a copy
    has_skels = (data_frame.body_pcd and data_frame.skel_frame and
                 data_frame.skel_frame.skeleton_list)

    # Merge body pcd points to scene pcd if body pcd exists
    if has_skels:
        body_pcd = o3d.geometry.PointCloud(data_frame.body_pcd)
        full_scene_pcd.points = o3d.utility.Vector3dVector(np.vstack((
            np.asarray(full_scene_pcd.points),
            np.asarray(body_pcd.points)
        )))
        full_scene_pcd.colors = o3d.utility.Vector3dVector(np.vstack((
            np.asarray(full_scene_pcd.colors),
            np.asarray(body_pcd.colors)
        )))
        full_scene_pcd.estimate_normals(fast_normal_computation=False)
        full_scene_pcd.orient_normals_towards_camera_location(data_frame.kinect_pose.trans_vec)

    return full_scene_pcd


def remove_human_points(scene_pcd: o3d_pcd, top_bb_list: List[o3d_bb],
                        bot_bb_list: List[o3d_bb]) -> o3d_pcd:
    """ Return the scene pointcloud with points within the top bounding box list, and
        the bottom bounding box list removed. All points from top bounding boxes are
        removed, irrespective of their normal directions; points from bottom bounding
        boxes are removed, but only for those with normals differ than that of the
        nominal ground (vertical).
    """
    # Remove points in the top_bb and bot_bb lists
    scene_pcd_no_man = o3d_pcd(scene_pcd)
    full_scene_points = np.asarray(scene_pcd_no_man.points)
    full_scene_normals = np.asarray(scene_pcd_no_man.normals)
    remove_pts_indices: np.ndarray = np.array([])

    # Mark every points from top_bb for removal
    for top_bb in top_bb_list:
        top_bb_pts_indices = top_bb.get_point_indices_within_bounding_box(scene_pcd_no_man.points)
        remove_pts_indices = np.append(remove_pts_indices, top_bb_pts_indices)
    
    # Mark points from bot_bb whose normal is not so vertical for removal
    for bot_bb in bot_bb_list:
        bot_bb_pts_indices = bot_bb.get_point_indices_within_bounding_box(scene_pcd_no_man.points)
        bot_bb_pts_sel = np.logical_and(
            np.isin(np.arange(len(full_scene_points)), bot_bb_pts_indices),
            full_scene_normals[:, 2] < Z_NORMAL_THOLD
        )
        remove_pts_indices = np.append(remove_pts_indices, np.arange(len(full_scene_points))[bot_bb_pts_sel])

    # Remove marked points from pcd
    remove_pts_indices = list(set(remove_pts_indices.astype(int).tolist()))
    full_scene_points[remove_pts_indices, 0] = np.nan
    scene_pcd_no_man.remove_non_finite_points()
    scene_pcd_no_man.estimate_normals(fast_normal_computation=False)
    scene_pcd_no_man.orient_normals_towards_camera_location(data_frame.kinect_pose.trans_vec)

    return scene_pcd_no_man


def process_data_frame(data_frame: PcdDataFrame, viewer=None) -> None:
    """ The main data frame processing function that extract human body bounding
        boxes, apply tracking on them, and store the state and the results
        in the global variables (track manager, and bounding box lists).
        Visualization can be optional turned on in debug mode.
    """
    # Make a copy of the full scene pcd including the body pcd for later use
    full_scene_pcd = recover_full_scene_pcd(data_frame)
    scene_pcd = o3d_pcd(full_scene_pcd)

    # Remove all points whose normals are close to upright direction
    normals = np.asarray(scene_pcd.normals)
    points = np.asarray(scene_pcd.points)
    points[normals[:, 2] >= Z_NORMAL_THOLD, 0] = np.nan
    scene_pcd.remove_non_finite_points()

    has_skels = (data_frame.body_pcd and data_frame.skel_frame and
                 data_frame.skel_frame.skeleton_list)

    # Add skeleton interpolated points
    if has_skels:
        skel_idx = 0
        skel_pts_size_list = []
        all_skel_pts = None
        for skel_idx in range(len(data_frame.skel_frame.skeleton_list)):
            skel_pts = data_frame.skel_frame.get_interpolated_skel_pts(skel_idx)
            skel_pts_size_list.append(len(skel_pts))
            if all_skel_pts is None:
                all_skel_pts = skel_pts
            else:
                all_skel_pts = np.vstack((all_skel_pts, skel_pts))
        if all_skel_pts is not None:
            scene_pcd.points = o3d.utility.Vector3dVector(np.vstack((
                np.asarray(scene_pcd.points),
                all_skel_pts
            )))
            scene_pcd.colors = o3d.utility.Vector3dVector(np.vstack((
                np.asarray(scene_pcd.colors),
                np.tile([1., 0.706, 0.], len(all_skel_pts)).reshape((len(all_skel_pts), 3))
            )))
            scene_pcd.estimate_normals(fast_normal_computation=False)
            scene_pcd.orient_normals_towards_camera_location(data_frame.kinect_pose.trans_vec)
    
    # Segment the pointcloud
    labels = np.array(scene_pcd.cluster_dbscan(eps=0.2, min_points=10, print_progress=True))
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    scene_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # Find out the skeleton clusters, and find their bounding box
    BB_OVERSEG_THOLD = 2.0
    skel_bb_list = []
    if has_skels:
        pt_idx = 0
        for i in range(len(skel_pts_size_list)):
            pt_idx = pt_idx + skel_pts_size_list[-(i+1)]
            skel_label = labels[-pt_idx]
            skel_pts = np.asarray(scene_pcd.points)[labels == skel_label]

            skel_pcd = o3d.geometry.PointCloud()
            skel_pcd.points = o3d.utility.Vector3dVector(skel_pts)

            bb = skel_pcd.get_axis_aligned_bounding_box()

            # If cluster dimension is greater than BB_DIM_THOLD, meaning over-segmented,
            # then we take skeleton bb
            if np.all(np.array(bb.get_extent()) < BB_OVERSEG_THOLD):
                bb.color = [0, 1, 0]
                skel_bb_list.append(bb)
            else:
                skel_idx = len(skel_pts_size_list) - 1 - i
                bb = data_frame.skel_frame.get_tight_skel_bounding_box(skel_idx)
                bb.color = [1, 0, 0]
                # TODO: find better ways to scale this bb along only the x, y axes
                skel_bb_list.append(bb.scale(1.1, bb.get_center()))
    
    # Maintaining glb_skel_bb_list
    glb_skel_bb_list.extend(skel_bb_list)

    # Tracking
    glb_track_man.update_tracks(data_frame.ts, skel_bb_list)
    #print(glb_track_man)

    # Each skel track will produce two BBs, top_bb and bot_bb, corresponding
    # to the top part and the bottom part of the skeleton. We make this
    # distinction because we do not want to falsely remove ground points
    # with close to vertical normals in top_bb, while we want to remove
    # all the points in the bot_bb.
    BOTTOM_BB_HEIGHT_THOLD = 0.3
    cur_top_bb_list = []
    cur_bot_bb_list = []
    track_list = glb_track_man.get_cur_tracks()
    if track_list:
        for trk in track_list:
            top_bb, bot_bb = trk.get_bb_pair(bottom_height=BOTTOM_BB_HEIGHT_THOLD, prescale=1.1)
            cur_top_bb_list.append(top_bb)
            cur_bot_bb_list.append(bot_bb)
    glb_tracked_top_bb_list.extend(cur_top_bb_list)
    glb_tracked_bot_bb_list.extend(cur_bot_bb_list)

    if viewer is None:
        return

    ################# Visualization #################
    show_bbs = True
    show_scene_pcd = False
    show_pcd_inside_bbs = True
    show_scene_pcd_excl_bbs = True
    show_bb_center_trace = True
    show_bb_trace = True
    show_track_bbs = True
    show_full_scene_pcd_no_man = False

    if show_bbs:
        for bb in skel_bb_list:
            viewer.add_geometry(bb)

    if show_scene_pcd:
        viewer.add_geometry(scene_pcd)

    if show_scene_pcd_excl_bbs:
        viewer.add_geometry(exclude_pts_in_bounding_boxes(scene_pcd, skel_bb_list))
    
    if show_pcd_inside_bbs:
        for bb in skel_bb_list:
            bb_pcd = get_sub_pcd_from_bounding_box(scene_pcd, bb, [1., 0.706, 0.])
            viewer.add_geometry(bb_pcd)
    
    if show_bb_center_trace and len(glb_skel_bb_list) > 0:
        bb_center_list = []
        for bb in glb_skel_bb_list:
            bb_center_list.append(bb.get_center())
        bb_center_trace_pcd = o3d.geometry.PointCloud()
        bb_center_trace_pcd.points = o3d.utility.Vector3dVector(np.array(bb_center_list))
        bb_center_trace_pcd.paint_uniform_color([0., 1.0, 0.])
        viewer.add_geometry(bb_center_trace_pcd)
    
    if show_bb_trace:
        for bb in glb_skel_bb_list:
            viewer.add_geometry(bb)
    
    if show_track_bbs:
        for bb in cur_top_bb_list + cur_bot_bb_list:
            bb.color = [1, 0.706, 0]
            viewer.add_geometry(bb)
    
    if show_full_scene_pcd_no_man:
        if cur_top_bb_list and cur_bot_bb_list:
            full_scene_pcd_no_man = remove_human_points(full_scene_pcd, cur_top_bb_list,
                                                        cur_bot_bb_list)
            viewer.add_geometry(full_scene_pcd_no_man)
        else:
            viewer.add_geometry(full_scene_pcd)

    
    # Adjust camera to kinect pose, and some angle/translation offset for better viewing
    vctl = viewer.get_view_control()
    vctl.set_lookat(data_frame.kinect_pose.trans_vec)
    vctl.set_front(np.matmul(data_frame.kinect_pose.get_rot_mat3x3(), [-1., 0.2, 0.2]))
    vctl.set_up(np.matmul(data_frame.kinect_pose.get_rot_mat3x3(), [0., 0., 1.]))
    vctl.translate(0., 50)
    vctl.set_zoom(0.2)


def view_data_frame(viewer, data_frame: PcdDataFrame):
    """ The main function to view the original pointclouds of the dataset,
        with viewing camera following the kinect camera pose.
    """
    # Add an coordinate system mesh at at the map origin
    viewer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.])))

    # Point clouds
    viewer.add_geometry(data_frame.pcd)
    if data_frame.body_pcd is not None:
        viewer.add_geometry(data_frame.body_pcd)
    
    # Kinect coordinate geometry
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=np.array([0., 0., 0.]))
    coord.transform(data_frame.kinect_pose.get_tf_mat4x4())
    viewer.add_geometry(coord)
    
    # Skeleton detection related
    if data_frame.skel_frame is not None:
        # Joint parts coordinate geometry
        for skel in data_frame.skel_frame.skeleton_list:
            for p in skel.skeleton_parts:
                coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=np.array([0., 0., 0.]))
                coord.transform(p.trans.get_tf_mat4x4())
                viewer.add_geometry(coord)
    
        # Create PointCloud connecting skeleton joints
        skel_idx = 0
        for skel_idx in range(len(data_frame.skel_frame.skeleton_list)):
            ext_skel_pcd = o3d.geometry.PointCloud()
            skel_pts = data_frame.skel_frame.get_interpolated_skel_pts(skel_idx)
            ext_skel_pcd.points = o3d.utility.Vector3dVector(skel_pts)
            ext_skel_pcd.paint_uniform_color([1., 0.706, 0.])
            viewer.add_geometry(ext_skel_pcd)

    # Adjust camera to kinect pose, and some angle/translation offset for better viewing
    vctl = viewer.get_view_control()
    vctl.set_lookat(data_frame.kinect_pose.trans_vec)
    vctl.set_front(np.matmul(data_frame.kinect_pose.get_rot_mat3x3(), [-1., 0.2, 0.2]))
    vctl.set_up(np.matmul(data_frame.kinect_pose.get_rot_mat3x3(), [0., 0., 1.]))
    vctl.translate(0., 50)
    vctl.set_zoom(0.2)


def view_result_pcd(viewer, data_frame: PcdDataFrame) -> None:
    """ View the result pointcloud with human removed.
    """
    # Add an coordinate system mesh at at the map origin
    viewer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.])))

    # Point clouds
    if data_frame.res_pcd:
        viewer.add_geometry(data_frame.res_pcd)
    
    # Kinect coordinate geometry
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=np.array([0., 0., 0.]))
    coord.transform(data_frame.kinect_pose.get_tf_mat4x4())
    viewer.add_geometry(coord)
    
    # Adjust camera to kinect pose, and some angle/translation offset for better viewing
    vctl = viewer.get_view_control()
    vctl.set_lookat(data_frame.kinect_pose.trans_vec)
    vctl.set_front(np.matmul(data_frame.kinect_pose.get_rot_mat3x3(), [-1., 0.2, 0.2]))
    vctl.set_up(np.matmul(data_frame.kinect_pose.get_rot_mat3x3(), [0., 0., 1.]))
    vctl.translate(0., 50)
    vctl.set_zoom(0.2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Human removal for Kinect 3D pointcloud sequence.')

    parser.add_argument('-b', '--batchmode',
                        help='Batch processing mode (default debug-visualization mode).',
                        action="store_true")
    parser.add_argument('-s', '--start',
                        help='Start data frame serial number (default 0) in debug-visualization mode',
                        type=int, default=0)
    parser.add_argument('dataset', help='Path to dataset.')

    args = parser.parse_args()

    print(f"batchmode: {args.batchmode}\n"
          f"start: {args.start}\n"
          f"dataset: {args.dataset}"
          )

    DBG_VIS = not args.batchmode
    
    cwd = os.getcwd()
    data_folder = os.path.join(os.getcwd(), args.dataset)
    if not os.path.isdir(data_folder):
        raise Exception(f"subfolder {data_folder} doesn't exist")
    
    try:
        pcd_data_files_list = get_pcd_data_files_list(data_folder)
    except Exception as e:
        print("Exception: " + e.args[0])
        sys.exit(1)
    
    data_frame_list: List[PcdDataFrame] = []
    for data_files_frame in pcd_data_files_list:
        data_frame = PcdDataFrame()
        data_frame.from_PcdDataFiles(data_files_frame)
        data_frame_list.append(data_frame)

    if DBG_VIS: # Debug and visualization mode
        viewer = o3d.visualization.VisualizerWithKeyCallback()
        viewer.create_window()
        opt = viewer.get_render_option()
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
    
        ts_index = args.start  # P1: 57, P2: 63, P3: 26
        def goto_next_frame(viewer):
            global ts_index
            if ts_index + 1 < len(data_frame_list):
                ts_index = ts_index + 1
                viewer.clear_geometries()
                view_data_frame(viewer, data_frame_list[ts_index])
    
        def goto_prev_frame(viewer):
            global ts_index
            if ts_index - 1 >= 0:
                ts_index = ts_index - 1
                viewer.clear_geometries()
                view_data_frame(viewer, data_frame_list[ts_index])
    
        def proc_cur_frame(viewer):
            global ts_index
            viewer.clear_geometries()
            process_data_frame(data_frame_list[ts_index], viewer=viewer)
        
        def view_res_pcd(viewer):
            global ts_index
            viewer.clear_geometries()
            view_result_pcd(viewer, data_frame_list[ts_index])
    
        viewer.register_key_callback(ord("."), goto_next_frame)
        viewer.register_key_callback(ord(","), goto_prev_frame)
        viewer.register_key_callback(ord("/"), proc_cur_frame)
        viewer.register_key_callback(ord("M"), view_res_pcd)

        view_data_frame(viewer, data_frame_list[ts_index])
        viewer.run()

    else:  # Batch processing mode
        all_top_bb_list = []
        all_bot_bb_list = []

        # Forward tracking and processing in time to get the top and bottom bb list
        for i in range(len(data_frame_list)):
            process_data_frame(data_frame_list[i])
        all_top_bb_list.extend(glb_tracked_top_bb_list)
        all_bot_bb_list.extend(glb_tracked_bot_bb_list)

        # Backward tracking and processing in time to get the top and bottom bb list
        glb_tracked_top_bb_list.clear()
        glb_tracked_bot_bb_list.clear()
        glb_track_man = tracking.TrackManager()
        for i in range(len(data_frame_list) - 1, -1, -1):
            process_data_frame(data_frame_list[i])
        all_top_bb_list.extend(glb_tracked_top_bb_list)
        all_bot_bb_list.extend(glb_tracked_bot_bb_list)

        # Put pointclouds with human removed in result folder
        res_folder = os.path.join(data_folder, "result")
        try:
            shutil.rmtree(res_folder)
        except Exception as e:
            pass
        os.mkdir(res_folder)

        for data_frame in data_frame_list:
            full_scene_pcd = recover_full_scene_pcd(data_frame)
            scene_pcd_no_human = remove_human_points(full_scene_pcd, all_top_bb_list,
                                                     all_bot_bb_list)
            filename = os.path.join(res_folder, f"{data_frame.ts:010d}.pcd")
            o3d.io.write_point_cloud(filename, scene_pcd_no_human)
