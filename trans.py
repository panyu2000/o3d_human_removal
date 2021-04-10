#!/home/panyu/Devs/miniconda3/bin/python

import numpy as np
import scipy.spatial.transform
import unittest

class Trans:
    def __init__(self):
        self.trans_vec = [0., 0., 0.]
        self.rot_quat = [0., 0., 0., 0.]
        self.rpy = [0., 0., 0.]
    
    def __str__(self):
        return (
            f"trans_vec: {self.trans_vec}, rot_quat: {self.rot_quat}, rpy: {self.rpy}"
        )
    
    def get_rot_mat3x3(self) -> np.ndarray:
        r = scipy.spatial.transform.Rotation.from_quat([self.rot_quat[1], self.rot_quat[2],
                                                        self.rot_quat[3], self.rot_quat[0]])
        return (r.as_matrix())
    
    def get_euler(self) -> np.ndarray:
        r = scipy.spatial.transform.Rotation.from_quat([self.rot_quat[1], self.rot_quat[2],
                                                        self.rot_quat[3], self.rot_quat[0]])
        return (r.as_euler('xyz', degrees=False))
    
    def get_tf_mat4x4(self) -> np.ndarray:
        rot3x3 = self.get_rot_mat3x3()
        rot4x4 = np.pad(rot3x3, ((0, 1), (0, 1)), mode='constant', constant_values=0.)
        rot4x4[0, 3] = self.trans_vec[0]
        rot4x4[1, 3] = self.trans_vec[1]
        rot4x4[2, 3] = self.trans_vec[2]
        rot4x4[3, 3] = 1.0
        return rot4x4
    
    def from_mat4x4(self, mat4x4: np.ndarray):
        r = scipy.spatial.transform.Rotation.from_matrix(mat4x4[0:3, 0:3])
        quat = r.as_quat()
        rpy = r.as_euler('xyz', degrees=False)
        self.trans_vec = [mat4x4[0, 3], mat4x4[1, 3], mat4x4[2, 3]]
        self.rot_quat = [quat[3], quat[0], quat[1], quat[2]]
        self.rpy = [rpy[0], rpy[1], rpy[2]]


class TestTrans(unittest.TestCase):
    def test_euler(self):
        trans = Trans()
        trans.rot_quat = [0.546847, 0.0608794, 0.086032, -0.830572]
        euler = trans.get_euler()
        self.assertTrue(np.allclose(euler, np.array([-0.0779044, 0.196484, -1.98476]), atol=1e-6))

    def test_back_forth(self):
        trans1 = Trans()
        trans1.trans_vec = [0.00718142, -0.000122707, -0.000580234]
        trans1.rot_quat = [0.995045, -0.0280831, 0.0919455, -0.0253696]
        trans1.rpy = [-0.0616155, 0.182567, -0.0566228]
        mat4x4 = trans1.get_tf_mat4x4()
        trans2 = Trans()
        trans2.from_mat4x4(mat4x4)

        self.assertTrue(
            np.allclose(trans1.trans_vec, trans2.trans_vec, atol=1e-6) and
            np.allclose(trans1.rot_quat, trans2.rot_quat, atol=1e-6) and
            np.allclose(trans1.rpy, trans2.rpy, atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
