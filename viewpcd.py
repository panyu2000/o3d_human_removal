#!/usr/bin/env python

import glob
import numpy as np
import open3d as o3d
import os

def view_pcd(viewer, pcd_file: str) -> None:
    pcd = o3d.io.read_point_cloud(pcd_file)

    # Add an coordinate system mesh at at the map origin
    viewer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.])))

    # Point clouds
    viewer.add_geometry(pcd)

    # Adjust camera to kinect pose, and some angle/translation offset for better viewing
    vctl = viewer.get_view_control()
    vctl.set_lookat([0, 0, 1.5])
    vctl.set_front([0.35, 0.88, 0.31])
    vctl.set_up([0., 0., 1.])
    vctl.translate(0., 0)
    vctl.set_zoom(0.3)


if __name__ == "__main__":
    cwd = os.getcwd()
    print(cwd)
    view_folder = os.path.join(os.getcwd(), "data/galleryP1/result")
    if not os.path.isdir(view_folder):
        raise Exception(f"subfolder {view_folder} doesn't exist")
    
    pcd_files = glob.glob(view_folder + "/*.pcd")
    pcd_files = list(filter(lambda f: os.path.isfile(f), pcd_files))
    pcd_files.sort()

    viewer = o3d.visualization.VisualizerWithKeyCallback()
    viewer.create_window()
    opt = viewer.get_render_option()
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    
    ts_index = 0
    def goto_next_frame(viewer):
        global ts_index
        if ts_index + 1 < len(pcd_files):
            ts_index = ts_index + 1
            viewer.clear_geometries()
            view_pcd(viewer, pcd_files[ts_index])
    
    def goto_prev_frame(viewer):
        global ts_index
        if ts_index - 1 >= 0:
            ts_index = ts_index - 1
            viewer.clear_geometries()
            view_pcd(viewer, pcd_files[ts_index])
    
    viewer.register_key_callback(ord("."), goto_next_frame)
    viewer.register_key_callback(ord(","), goto_prev_frame)

    view_pcd(viewer, pcd_files[ts_index])
    viewer.run()