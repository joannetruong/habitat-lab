import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import squaternion
from habitat.utils.geometry_utils import quaternion_rotate_vector, quaternion_from_coeff
from habitat.tasks.utils import cartesian_to_polar
import magnum as mn


# File I/O related
def parse_config(config):
    with open(config, 'r') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    return config_data


# Geometry related
def rotate_vector_3d(v, r, p, y):
    """Rotates 3d vector by roll, pitch and yaw counterclockwise"""
    local_to_global = R.from_euler('xyz', [r, p, y]).as_dcm()
    global_to_local = local_to_global.T
    return np.dot(global_to_local, v)


def scalar_vector_to_quat(scalar, vector):
    new_scalar = np.cos(scalar / 2)
    new_vector = np.array(vector) * np.sin(scalar / 2)
    quat = squaternion.Quaternion(new_scalar, *new_vector)
    return quat


def quaternion_from_coeff(coeffs: np.ndarray) -> np.quaternion:
    r"""Creates a quaternions from coeffs in [x, y, z, w] format"""
    quat = np.quaternion(0, 0, 0, 0)
    quat.real = coeffs[3]
    quat.imag = coeffs[0:3]
    return quat


def quaternion_rotate_vector(quat: np.quaternion, v: np.array) -> np.array:
    r"""Rotates a vector by a quaternion
    Args:
        quaternion: The quaternion to rotate by
        v: The vector to rotate
    Returns:
        np.array: The rotated vector
    """
    vq = np.quaternion(0, 0, 0, 0)
    vq.imag = v
    return (quat * vq * quat.inverse()).imag


def cartesian_to_polar(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def get_rpy(rotation, transform=True):
    obs_quat = squaternion.Quaternion(rotation.scalar, *rotation.vector)
    if transform:
        inverse_base_transform = scalar_vector_to_quat(np.pi / 2, (1, 0, 0))
        obs_quat = obs_quat * inverse_base_transform
    roll, yaw, pitch = obs_quat.to_euler()
    return np.array([roll, pitch, yaw])


def quat_from_magnum(quat: mn.Quaternion) -> np.quaternion:
    a = np.quaternion(1, 0, 0, 0)
    a.real = quat.scalar
    a.imag = quat.vector
    return a


def quat_to_rad(rotation):
    rot = quat_from_magnum(rotation)

    if isinstance(rotation, list):
        rot = quaternion_from_coeff(np.array(rot))
    heading_vector = quaternion_rotate_vector(
        rot.inverse(), np.array([0, 0, -1])
    )

    # r,y,p = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return heading_vector


def euler_from_quaternion(quat):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """

    w = quat.scalar
    x = quat.vector.x
    y = quat.vector.y
    z = quat.vector.z
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return np.array([roll_x, pitch_y, yaw_z])  # in radians


def rotate_vector_2d(v, yaw):
    """Rotates 2d vector by yaw counterclockwise"""
    local_to_global = R.from_euler('z', yaw).as_dcm()
    global_to_local = local_to_global.T
    global_to_local = global_to_local[:2, :2]
    if len(v.shape) == 1:
        return np.dot(global_to_local, v)
    elif len(v.shape) == 2:
        return np.dot(global_to_local, v.T).T
    else:
        print('Incorrect input shape for rotate_vector_2d', v.shape)
        return v


def l2_distance(v1, v2):
    """Returns the L2 distance between vector v1 and v2."""
    return np.linalg.norm(np.array(v1) - np.array(v2))


def cartesian_to_polar(x, y):
    """Convert cartesian coordinate to polar coordinate"""
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def quatFromXYZW(xyzw, seq):
    """Convert quaternion from XYZW (pybullet convention) to arbitrary sequence."""
    assert len(seq) == 4 and 'x' in seq and 'y' in seq and 'z' in seq and 'w' in seq, \
        "Quaternion sequence {} is not valid, please double check.".format(seq)
    inds = ['xyzw'.index(axis) for axis in seq]
    return xyzw[inds]


def quatToXYZW(orn, seq):
    """Convert quaternion from arbitrary sequence to XYZW (pybullet convention)."""
    assert len(seq) == 4 and 'x' in seq and 'y' in seq and 'z' in seq and 'w' in seq, \
        "Quaternion sequence {} is not valid, please double check.".format(seq)
    inds = [seq.index(axis) for axis in 'xyzw']
    return orn[inds]