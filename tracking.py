#!/home/panyu/Devs/miniconda3/bin/python

import copy
import math
import numpy as np
import open3d as o3d
import pdb
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple
import unittest

Vec3 = Tuple[float, float, float]
o3d_bb = o3d.geometry.AxisAlignedBoundingBox

class KalmanConstVel:
    INIT_SIGMA = 1000.  # initial sigma for pos and vel
    ACCEL_HORZ_SIGMA = 2.  # std acceleration along x, y axes
    ACCEL_VERT_SIGMA = 0.5 # std acceleration along z axis, smaller than that of horz
    OBSV_HORZ_SIGMA = 0.2  # std observation noise along x, y axes
    OBSV_VERT_SIGMA = 0.3  # std observation noise along z axis, bigger than that of horz

    def __init__(self):
        self.X = np.zeros((6, 1))  # x, y, z, vx, vy, vz
        self.P = np.eye(6) * self.INIT_SIGMA**2
    
    def __str__(self) -> str:
        with np.printoptions(precision=6, suppress=True):
            return (
                f"X: {self.X}\n"
                f"P: {self.P}"
            )
    
    def propagate(self, dt: float) -> None:
        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4
        A = np.array([
            [1, 0, 0, dt,  0,  0],
            [0, 1, 0,  0, dt,  0],
            [0, 0, 1,  0,  0, dt],
            [0, 0, 0,  1,  0,  0],
            [0, 0, 0,  0,  1,  0],
            [0, 0, 0,  0,  0,  1]
        ])
        Q = np.array([
            [dt4/4,      0,    0, dt3/2,     0,     0],
            [    0, dt4/4,     0,     0, dt3/2,     0],
            [    0,     0, dt4/4,     0,     0, dt3/2],
            [dt3/2,     0,     0,   dt2,     0,     0],
            [    0, dt3/2,     0,     0,   dt2,     0],
            [    0,     0, dt3/2,     0,     0,   dt2]
        ])
        Q[(0,3), :] *= self.ACCEL_HORZ_SIGMA**2
        Q[(1,4), :] *= self.ACCEL_HORZ_SIGMA**2
        Q[(2,5), :] *= self.ACCEL_VERT_SIGMA**2

        self.X = A @ self.X
        self.P = A @ self.P @ A.T + Q
    
    def update(self, meas: Vec3) -> None:
        """ Update with measurement of [x, y, z] location
        """
        z = np.array(meas).reshape((3,1))
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ])
        R = np.diag([self.OBSV_HORZ_SIGMA**2, self.OBSV_HORZ_SIGMA**2, self.OBSV_VERT_SIGMA**2])
        y = z - H @ self.X
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.pinv(S)
        self.X = self.X + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P


class DataAssoResult:
    def __init__(self):
        self.matched_indices_pairs: List[Tuple[int, int]] = []
        self.unmatched_track_indices: List[int] = []
        self.unmatched_meas_indices: List[int] = []
        self.cost_mat: np.ndarray = np.array([])
    
    def __str__(self) -> str:
        with np.printoptions(precision=3, suppress=True):
            return (
                f"matched_indices_pairs: {self.matched_indices_pairs}\n"
                f"unmatched_track_indices: {self.unmatched_track_indices}\n"
                f"unmatched_meas_indices: {self.unmatched_meas_indices}\n"
                f"cost_mat: {self.cost_mat}"
            )


# Data association with naive cost function using euclidean distance
def data_association(track_centers: List[Vec3], meas_centers: List[Vec3]) -> DataAssoResult:
    res = DataAssoResult()

    if not track_centers and not meas_centers:
        return res
    elif not track_centers:
        res.unmatched_meas_indices = list(range(len(meas_centers)))
        return res
    elif not meas_centers:
        res.unmatched_track_indices = list(range(len(track_centers)))
        return res

    num_tracks = len(track_centers)
    num_meas = len(meas_centers)
    
    cost_mat = np.zeros((num_tracks, num_meas))
    i, j = 0, 0
    for i in range(num_tracks):
        for j in range(num_meas):
            dx = track_centers[i][0] - meas_centers[j][0]
            dy = track_centers[i][1] - meas_centers[j][1]
            dz = track_centers[i][2] - meas_centers[j][2]
            cost_mat[i, j] = math.hypot(dx, dy, dz)
    
    row_indices, col_indices = linear_sum_assignment(cost_mat)

    res.matched_indices_pairs = list(zip(row_indices, col_indices))
    res.unmatched_track_indices = list(set(range(num_tracks)).difference(row_indices))
    res.unmatched_meas_indices = list(set(range(num_meas)).difference(col_indices))
    res.cost_mat = cost_mat

    return res


class SkelTrack:
    def __init__(self):
        self.is_initialized: bool = False
        self.track_id: int = 0
        self.last_propagated_ts: int = 0
        self.last_updated_ts: int = 0
        self.kf: KalmanConstVel = None
        self.meas_bb_history: List[Tuple[int, o3d_bb]] = []
    
    def __str__(self) -> str:
        return (
            f"is_initialized: {self.is_initialized}\n"
            f"track_id: {self.track_id}\n"
            f"last_propagated_ts: {self.last_propagated_ts}\n"
            f"last_updated_ts: {self.last_updated_ts}\n"
            f"kf: {self.kf}\n"
        )
    
    def initialize(self, track_id: int, ts: int, bb: o3d_bb) -> None:
        self.track_id = track_id
        self.kf = KalmanConstVel()
        self.kf.update(bb.get_center())
        self.last_propagated_ts = ts
        self.last_updated_ts = ts
        self.meas_bb_history.append((ts, bb))
        self.is_initialized = True
    
    def predict(self, ts: int) -> None:
        assert(self.is_initialized)
        if ts != self.last_propagated_ts:
            dt = (ts - self.last_propagated_ts) / 1e6
            self.kf.propagate(dt)
            self.last_propagated_ts = ts

    def update(self, ts: int, bb: o3d_bb) -> None:
        assert(self.is_initialized)
        if self.last_propagated_ts != ts:
            self.predict(ts)
        self.kf.update(bb.get_center())
        self.last_updated_ts = ts
        self.meas_bb_history.append((ts, bb))

    def get_motion_state(self) -> List[float]:
        assert(self.is_initialized)
        return self.kf.X.reshape(6).tolist()
    
    def get_pos(self) -> List[float]:
        return self.get_motion_state()[0:3]
    
    def get_vel(self) -> List[float]:
        return self.get_motion_state()[3:6]

    def get_extent(self) -> List[float]:
        assert(self.is_initialized)
        LAST_N = 5
        cut_bb_hist = self.meas_bb_history[-LAST_N:]
        sum_x = sum_y = sum_z = 0
        for _, bb in cut_bb_hist:
            extent = bb.get_extent()
            sum_x += extent[0]
            sum_y += extent[1]
            sum_z += extent[2]
        n = len(cut_bb_hist)
        return [sum_x/n, sum_y/n, sum_z/n]
        
    def get_propagated_dt(self) -> float:
        return (self.last_propagated_ts - self.last_updated_ts) / 1e6

    def get_bb(self) -> o3d.geometry.AxisAlignedBoundingBox:
        assert(self.is_initialized)
        cx, cy, cz = self.get_pos()
        dx, dy, dz = np.array(self.get_extent()) / 2
        return o3d.geometry.AxisAlignedBoundingBox(min_bound=[cx-dx, cy-dy, cz-dz],
                                                   max_bound=[cx+dx, cy+dy, cz+dz])
    
    def get_bb_pair(self, bottom_height:float, prescale: float=1) -> Tuple[o3d_bb, o3d_bb]:
        """ Prescale the extent, and return a bottom bb and a top bb, broken up by bottom height
        """
        assert(self.is_initialized)
        cx, cy, cz = self.get_pos()
        dx, dy, dz = np.array(self.get_extent()) / 2 * prescale
        has_top_bb = bottom_height < dz*2
        top_bb = o3d_bb(min_bound=[cx-dx, cy-dy, cz-dz+bottom_height],
                        max_bound=[cx+dx, cy+dy, cz+dz]) if has_top_bb else o3d_bb()
        bot_bb = o3d_bb(min_bound=[cx-dx, cy-dy, cz-dz],
                        max_bound=[cx+dx, cy+dy, cz-dz+bottom_height])
        return (top_bb, bot_bb)


class TrackManager:
    DATA_ASSO_GATE_DIST = 1.2  # in meters
    TRACK_EXPIRE_GATE_TIME = 3  # in seconds

    def __init__(self):
        # All tracks = cur_tracks + dormant_tracks
        self.cur_tracks: List[SkelTrack] = []
        self.past_tracks: List[SkelTrack] = []
        self.last_ts: int = 0
        self.next_track_id: int = 0

        # Below are different views of tracks in cur_tracks and dormant_tracks
        self.propagated_tracks: List[SkelTrack] = []
        self.updated_tracks: List[SkelTrack] = []
        self.new_born_tracks: List[SkelTrack] = []
        self.new_dead_tracks: List[SkelTrack] = []

    def __str__(self) -> str:
        def T(l: List[float]) -> str:
            return f"[{l[0]:.2f}, {l[1]:.2f}, {l[2]:.2f}]"
        ret = "===== cur_tracks =====\n"
        for trk in self.cur_tracks:
            ret += f"  id: {trk.track_id}, pos: {T(trk.get_pos())}, vel: {T(trk.get_vel())}, ext: {T(trk.get_extent())}\n"
        del T
        ret += "\n===== past_tracks =====\n"
        for trk in self.past_tracks:
            ret += f"{trk.track_id}, "
        ret += "\n===== propagated_tracks =====\n"
        for trk in self.propagated_tracks:
            ret += f"{trk.track_id}, "
        ret += "\n===== updated_tracks =====\n"
        for trk in self.updated_tracks:
            ret += f"{trk.track_id}, "
        ret += "\n===== new_born_tracks =====\n"
        for trk in self.new_born_tracks:
            ret += f"{trk.track_id}, "
        ret += "\n===== new_dead_tracks =====\n"
        for trk in self.new_dead_tracks:
            ret += f"{trk.track_id}, "
        ret += "\n"
        ret += (
            f"last_ts: {self.last_ts}\n"
            f"next_track_id: {self.next_track_id}\n"
        )
        return ret

    def get_cur_tracks(self) -> List[SkelTrack]:
        return copy.deepcopy(self.cur_tracks)
    
    def update_tracks(self, ts: int, meas_bb_list: List[o3d_bb]) -> None:
        self.last_ts = ts
        self.propagated_tracks.clear()
        self.updated_tracks.clear()
        self.new_born_tracks.clear()
        self.new_dead_tracks.clear()

        if not self.cur_tracks and not meas_bb_list:
            return

        meas_centers = []
        for meas_bb in meas_bb_list:
            meas_centers.append(meas_bb.get_center().tolist())
        
        # Propagate tracks
        track_centers = []
        for trk in self.cur_tracks:
            trk.predict(ts)
            track_centers.append(trk.get_pos())

        # If has measurement
        if meas_centers:
            data_asso_res = data_association(track_centers, meas_centers)

            # Update tracks with paired measurements
            for trk_idx, meas_idx in data_asso_res.matched_indices_pairs:
                dx = track_centers[trk_idx][0] - meas_centers[meas_idx][0]
                dy = track_centers[trk_idx][1] - meas_centers[meas_idx][1]
                dz = track_centers[trk_idx][2] - meas_centers[meas_idx][2]

                # if matched pair is faraway, break them apart
                if math.hypot(dx, dy, dz) > self.DATA_ASSO_GATE_DIST:
                    data_asso_res.unmatched_track_indices.append(trk_idx)
                    data_asso_res.unmatched_meas_indices.append(meas_idx)
                else:
                    self.cur_tracks[trk_idx].update(ts, meas_bb_list[meas_idx])
                    self.updated_tracks.append(self.cur_tracks[trk_idx])
            
            # Kill tracks that propagated for too long
            tmp_cur_tracks: List[SkelTrack] = []
            for trk in self.cur_tracks:
                propagated_dt = abs(trk.get_propagated_dt())
                if propagated_dt < self.TRACK_EXPIRE_GATE_TIME:
                    tmp_cur_tracks.append(trk)
                else:
                    self.past_tracks.append(trk)
                    self.new_dead_tracks.append(trk)
                if propagated_dt != 0:
                    self.propagated_tracks.append(trk)
            self.cur_tracks = tmp_cur_tracks

            # Spawn new tracks from unmatched measurements
            for meas_idx in data_asso_res.unmatched_meas_indices:
                new_track = SkelTrack()
                new_track.initialize(self.next_track_id, ts, meas_bb_list[meas_idx])
                self.next_track_id += 1
                self.cur_tracks.append(new_track)
                self.new_born_tracks.append(new_track)
        
        else:
            # No measurement, kill tracks that propagated for too long
            tmp_cur_tracks: List[SkelTrack] = []
            for trk in self.cur_tracks:
                propagated_dt = abs(trk.get_propagated_dt())
                if propagated_dt < self.TRACK_EXPIRE_GATE_TIME:
                    tmp_cur_tracks.append(trk)
                else:
                    self.past_tracks.append(trk)
                    self.new_dead_tracks.append(trk)
                if propagated_dt != 0:
                    self.propagated_tracks.append(trk)
            self.cur_tracks = tmp_cur_tracks
            

class TestDataAsso(unittest.TestCase):
    def test_no_track_or_no_meas(self):
        print("Test with no tracks or no meas")
        track_centers = [[0, 0, 0], [1, 1, 1], [-1, -1, -1]]
        meas_centers = [[1.1, 1.2, 1.], [-1.0, -1.1, -1.2], [0.3, 0.2, 0.1]]

        res = data_association([], [])
        self.assertTrue(not res.matched_indices_pairs)
        self.assertTrue(not res.unmatched_track_indices)
        self.assertTrue(not res.unmatched_meas_indices)
        self.assertTrue(res.cost_mat.size == 0)

        res = data_association([], meas_centers)
        self.assertTrue(not res.matched_indices_pairs)
        self.assertTrue(not res.unmatched_track_indices)
        self.assertTrue(set(res.unmatched_meas_indices) == {0, 1, 2})
        self.assertTrue(res.cost_mat.size == 0)
        
        res = data_association(track_centers, [])
        self.assertTrue(not res.matched_indices_pairs)
        self.assertTrue(set(res.unmatched_track_indices) == {0, 1, 2})
        self.assertTrue(not res.unmatched_meas_indices)
        self.assertTrue(res.cost_mat.size == 0)


    def test_square_cost_mat(self):
        print("Test square cost matrix")
        track_centers = [[0, 0, 0], [1, 1, 1], [-1, -1, -1]]
        meas_centers = [[1.1, 1.2, 1.], [-1.0, -1.1, -1.2], [0.3, 0.2, 0.1]]
        res = data_association(track_centers, meas_centers)
        self.assertTrue(set([(2, 1), (1, 0), (0, 2)]) == set(res.matched_indices_pairs))
        self.assertTrue(not res.unmatched_track_indices)
        self.assertTrue(not res.unmatched_meas_indices)
    

    def test_rectangular_cost_mat_with_more_tracks(self):
        print("Test rectangular cost matrix with more tracks")
        track_centers = [[0, 0, 0], [1, 1, 1], [-1, -1, -1], [3, 3, 3]]
        meas_centers = [[1.1, 1.2, 1.], [-1.0, -1.1, -1.2], [0.3, 0.2, 0.1]]
        res = data_association(track_centers, meas_centers)
        self.assertTrue(set([(2, 1), (1, 0), (0, 2)]) == set(res.matched_indices_pairs))
        self.assertTrue(set(res.unmatched_track_indices) == set([3]))
        self.assertTrue(not res.unmatched_meas_indices)


    def test_rectangular_cost_mat_with_more_meas(self):
        print("Test rectangular cost matrix with more measurements")
        track_centers = [[0, 0, 0], [1, 1, 1], [-1, -1, -1]]
        meas_centers = [[1.1, 1.2, 1.], [-1.0, -1.1, -1.2], [0.3, 0.2, 0.1], [0., 0.1, 0.]]
        res = data_association(track_centers, meas_centers)
        self.assertTrue(set([(2, 1), (1, 0), (0, 3)]) == set(res.matched_indices_pairs))
        self.assertTrue(not res.unmatched_track_indices)
        self.assertTrue(set(res.unmatched_meas_indices) == set([2]))


class TestKalmanConstVel(unittest.TestCase):
    def test_exercise_filter(self):
        meas_list = [[1.05, 0.95, 1.05], [2.0, 2.05, 1.95], [3.01, 2.98, 2.99]]
        kf = KalmanConstVel()
        # initialize with first meas
        kf.update(meas_list[0])

        # propagate and update with rest meas
        for meas in meas_list[1:]:
            kf.propagate(dt=1)
            kf.update(meas)
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
