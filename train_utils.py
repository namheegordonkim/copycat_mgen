import glob
import json
import zipfile
from typing import Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset, Sampler


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon
        self.epsilon = epsilon

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize(self, arr: np.ndarray) -> np.ndarray:
        return np.clip(
            (arr - self.mean) / np.sqrt(self.var + self.epsilon), -1000, 1000
        )

    def unnormalize(self, arr: np.ndarray) -> np.ndarray:
        return arr * np.sqrt(self.var + self.epsilon) + self.mean


class ThroughDataset(Dataset):
    """
    Sacrifice some readability to make life easier.
    Whatever input array/argument tensor provided will be the output for dataset.
    """

    def __init__(self, *args):
        self.args = args
        for a1, a2 in zip(self.args, self.args[1:]):
            assert a1.shape[0] == a2.shape[0]

    def __getitem__(self, index):
        indexed = tuple(torch.as_tensor(a[index]) for a in self.args)
        return indexed

    def __len__(self):
        return self.args[0].shape[0]


class RepeatSampler(Sampler):
    """Sampler that repeats samples to match a desired batch size."""

    def __init__(self, data_source, batch_size):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        indices = torch.randint(
            0, len(self.data_source), (self.batch_size,), dtype=torch.long
        )
        return iter(indices)

    def __len__(self):
        return len(self.data_source)


def quat_to_expm(quat: np.ndarray, eps: float = 1e-8):
    """
    Quaternion is (x, y, z, w)
    """
    im = quat[..., :3]
    im_norm = np.linalg.norm(im, axis=-1)
    half_angle = np.arctan2(im_norm, quat[..., 3])
    expm = np.where(
        im_norm[..., None] < eps,
        im,
        half_angle[..., None] * (im / im_norm[..., None]),
    )
    return expm


def quat_to_sixd(quat: np.ndarray):
    """
    Quaternion is (x, y, z, w)
    """
    rot = Rotation.from_quat(quat)
    sixd = rot.as_matrix()[..., :2].swapaxes(-1, -2).reshape(*quat.shape[:-1], 6)
    return sixd


def quat_to_sixd_torch(quat: torch.Tensor):
    """
    Quaternion is (x, y, z, w)
    """
    rotmat = rot_matrix_from_quaternion(quat)
    sixd = rotmat[..., :2].swapaxes(-1, -2).reshape(*quat.shape[:-1], 6)
    return sixd


def sixd_to_quat(sixd: np.ndarray):
    """
    Quaternion is (x, y, z, w)
    """
    sixd_reshaped = sixd.reshape((-1, 2, 3)).swapaxes(-1, -2)
    third_column = np.cross(sixd_reshaped[..., 0], sixd_reshaped[..., 1], axis=-1)
    rotmat = np.concatenate([sixd_reshaped, third_column[..., None]], axis=-1)
    rot = Rotation.from_matrix(rotmat)
    quat = rot.as_quat().reshape(*sixd.shape[:-1], 4)
    return quat


def quat_from_rotation_matrix(m):
    """
    Construct a 3D rotation from a valid 3x3 rotation matrices.
    Reference can be found here:
    http://www.cg.info.hiroshima-cu.ac.jp/~miyazaki/knowledge/teche52.html

    :param m: 3x3 orthogonal rotation matrices.
    :type m: Tensor

    :rtype: Tensor
    """
    diag0 = m[..., 0, 0]
    diag1 = m[..., 1, 1]
    diag2 = m[..., 2, 2]

    # Math stuff.
    w = (((diag0 + diag1 + diag2 + 1.0) / 4.0).clamp(0.0, None)) ** 0.5
    x = (((diag0 - diag1 - diag2 + 1.0) / 4.0).clamp(0.0, None)) ** 0.5
    y = (((-diag0 + diag1 - diag2 + 1.0) / 4.0).clamp(0.0, None)) ** 0.5
    z = (((-diag0 - diag1 + diag2 + 1.0) / 4.0).clamp(0.0, None)) ** 0.5

    # Only modify quaternions where w > x, y, z.
    c0 = (w >= x) & (w >= y) & (w >= z)
    x[c0] *= (m[..., 2, 1][c0] - m[..., 1, 2][c0]).sign()
    y[c0] *= (m[..., 0, 2][c0] - m[..., 2, 0][c0]).sign()
    z[c0] *= (m[..., 1, 0][c0] - m[..., 0, 1][c0]).sign()

    # Only modify quaternions where x > w, y, z
    c1 = (x >= w) & (x >= y) & (x >= z)
    w[c1] *= (m[..., 2, 1][c1] - m[..., 1, 2][c1]).sign()
    y[c1] *= (m[..., 1, 0][c1] + m[..., 0, 1][c1]).sign()
    z[c1] *= (m[..., 0, 2][c1] + m[..., 2, 0][c1]).sign()

    # Only modify quaternions where y > w, x, z.
    c2 = (y >= w) & (y >= x) & (y >= z)
    w[c2] *= (m[..., 0, 2][c2] - m[..., 2, 0][c2]).sign()
    x[c2] *= (m[..., 1, 0][c2] + m[..., 0, 1][c2]).sign()
    z[c2] *= (m[..., 2, 1][c2] + m[..., 1, 2][c2]).sign()

    # Only modify quaternions where z > w, x, y.
    c3 = (z >= w) & (z >= x) & (z >= y)
    w[c3] *= (m[..., 1, 0][c3] - m[..., 0, 1][c3]).sign()
    x[c3] *= (m[..., 2, 0][c3] + m[..., 0, 2][c3]).sign()
    y[c3] *= (m[..., 2, 1][c3] + m[..., 1, 2][c3]).sign()

    quat = torch.stack([x, y, z, w], dim=-1)
    return quat / torch.norm(quat, dim=-1, keepdim=True)


def sixd_to_quat_torch(sixd: torch.Tensor):
    """
    Quaternion is (x, y, z, w)
    """
    sixd_reshaped = sixd.reshape((*sixd.shape[:-1], 2, 3)).swapaxes(-1, -2)
    third_column = torch.cross(sixd_reshaped[..., 0], sixd_reshaped[..., 1], dim=-1)
    rotmat = torch.cat([sixd_reshaped, third_column[..., None]], dim=-1)
    quat = quat_from_rotation_matrix(rotmat)
    return quat


def rot_matrix_from_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def quat_to_expm_torch(quat: torch.Tensor, eps: float = 1e-8):
    """
    Quaternion is (x, y, z, w)
    """
    im = quat[..., :3]
    im_norm = torch.norm(im, dim=-1)
    half_angle = torch.arctan2(im_norm, quat[..., 3])
    expm = torch.where(
        im_norm[..., None] < eps,
        im,
        half_angle[..., None] * (im / im_norm[..., None]),
    )
    return expm


def expm_to_quat(expm: np.ndarray, eps: float = 1e-8):
    """ """
    half_angle = np.linalg.norm(expm, axis=-1)[..., None]
    # if half_angle < eps:
    #     quat = np.concatenate([expm, np.ones_like(half_angle)], axis=-1)
    #     quat /= np.linalg.norm(quat, axis=-1)[..., None]
    #     return quat
    c = np.cos(half_angle)
    s = np.sin(half_angle)
    quat = np.where(
        half_angle < eps,
        np.concatenate([expm, np.ones_like(half_angle)], axis=-1),
        np.concatenate([expm * s / half_angle, c], axis=-1),
    )
    return quat


def expm_to_quat_torch(expm: torch.Tensor, eps: float = 1e-8):
    """ """
    half_angle = torch.linalg.norm(expm, dim=-1)[..., None]
    c = torch.cos(half_angle)
    s = torch.sin(half_angle)
    quat = torch.where(
        half_angle < eps,
        torch.cat([expm, torch.ones_like(half_angle)], dim=-1),
        torch.cat([expm * s / half_angle, c], dim=-1),
    )
    return quat


def get_pos_expm(frames_np):
    timestamps = frames_np[1:, ..., 0]
    part_data = frames_np[1:, ..., 1:]
    part_data = part_data.reshape(part_data.shape[0], 3, -1)
    my_pos = part_data[..., :3]
    my_quat = part_data[..., 3:]
    my_rot = Rotation.from_quat(my_quat.reshape(-1, 4))
    correction = Rotation.from_euler("X", 0.5 * np.pi)
    correction_matrix = correction.as_matrix()
    my_pos = np.einsum("nkj,ij->nki", my_pos, correction_matrix)
    my_rot = correction * my_rot
    correction = Rotation.from_euler("Z", 0.5 * np.pi)
    correction_matrix = correction.as_matrix()
    my_pos = np.einsum("nkj,ij->nki", my_pos, correction_matrix)
    my_rot = correction * my_rot
    my_rotmat = my_rot.as_matrix().reshape((my_quat.shape[0], 3, 3, 3))
    my_rotmat[..., [0, 1, 2]] = my_rotmat[..., [2, 0, 1]]
    my_rotmat[:, [0, 1, 2]] = my_rotmat[:, [0, 2, 1]]
    my_rotmat[:, 1:, ..., [0, 1, 2]] = my_rotmat[:, 1:, ..., [0, 2, 1]]
    my_rotmat[:, 1:, ..., 2] *= -1
    my_rotmat[:, 1, ..., [1, 2]] *= -1

    # left_correction_rotmat = Rotation.from_euler("X", -0.5 * np.pi)
    # right_correction_rotmat = Rotation.from_euler("X", 0.5 * np.pi)
    # my_rotmat[:, 1] = np.einsum("ik, nkj -> nij", left_correction_rotmat.as_matrix(), my_rotmat[:, 1])
    # my_rotmat[:, 2] = np.einsum("ik, nkj -> nij", right_correction_rotmat.as_matrix(), my_rotmat[:, 2])
    # my_rotmat[:, 1, ..., [1, 2]] *= -1
    my_rot = Rotation.from_matrix(my_rotmat.reshape((-1, 3, 3)))
    my_quat = my_rot.as_quat().reshape(*my_quat.shape)
    my_sixd = my_rot.as_matrix()[..., :2].reshape(*my_quat.shape[:-1], 6)
    my_pos[..., [0, 1]] -= my_pos[0][0, [0, 1]]
    my_pos[:, [0, 1, 2]] = my_pos[:, [0, 2, 1]]
    my_pos_sixd = np.concatenate([my_pos, my_sixd], axis=-1)
    my_pos_sixd = my_pos_sixd.reshape(-1, 27)
    my_expm = quat_to_expm(my_quat)
    my_pos_expm = np.concatenate([my_pos, my_expm], axis=-1)
    my_pos_expm = my_pos_expm.reshape(-1, 18)
    return my_pos_expm, my_pos_sixd, timestamps


def open_boxrr_zip(zip_path: str, xror):
    level_filename = f'{xror.data["info"]["software"]["activity"]["difficulty"]}{xror.data["info"]["software"]["activity"]["mode"]}.dat'
    info_filename = "Info.dat"
    with zipfile.ZipFile(zip_path) as zipf:
        try:
            with zipf.open(level_filename) as f:
                song = json.load(f)
        except KeyError:
            level_filename = level_filename.replace("Standard", "")
            try:
                with zipf.open(level_filename) as f:
                    song = json.load(f)
            except KeyError as e:
                raise e

        try:
            with zipf.open(info_filename) as f:
                song_info = json.load(f)
        except KeyError:
            info_filename = "info.dat"
            try:
                with zipf.open(info_filename) as f:
                    song_info = json.load(f)
            except KeyError as e:
                raise e
    return song, song_info


def copycat_collate_fn(batch):
    n_sequences = len(batch)
    dd = {}
    for k in list(batch[0].keys())[1:-1]:
        dd[k] = []
        for d in batch:
            dd[k].append(d[k])

        # Apply nanpads
        all_lengths = torch.tensor([a.shape[0] for a in dd[k]])
        max_seq_len = all_lengths.max()
        lengths_to_go = max_seq_len - all_lengths

        nanpads = [
            torch.ones((lengths_to_go[i], *(dd[k][i].shape[1:]))) * torch.nan
            for i in range(len(batch))
        ]
        padded_tensors = [
            torch.cat([a, nanpads[i]], dim=0) for i, a in enumerate(dd[k])
        ]
        stacked_tensors = torch.stack(padded_tensors, dim=0)
        dd[k] = stacked_tensors
    dd["lengths"] = all_lengths
    return dd


# @torch.jit.script
def quat_inv(_q):
    q = _q.clone()
    q[..., :-1] = -1 * q[..., :-1]
    return q / torch.sum(q**2, dim=-1, keepdim=True)


# @torch.jit.script
def quat_rotate(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[..., :-1]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[..., -1:] * uv + uuv)).view(original_shape)


# @torch.jit.script
def quat_mul(a, b):
    """
    quaternion multiplication
    """
    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    q = torch.stack([x, y, z, w], dim=-1)
    return q / torch.norm(q, dim=-1, keepdim=True)
