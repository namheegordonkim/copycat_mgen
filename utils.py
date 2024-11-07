import os

import joblib

from poselib.skeleton.skeleton3d import SkeletonState
from pyvista_imgui import ImguiPlotter
from scipy.spatial.transform import Rotation
from typing import List

from train_utils import quat_to_expm
from visual_data import XMLVisualDataContainer
import numpy as np
import pyvista as pv


def load_skel(mjcf_path: str) -> List[pv.PolyData]:
    """
    Loads the skeleton mesh from the MJCF file.

    Args:
        mjcf_path (str): Path to the MJCF file.

    Returns:
        List[pv.PolyData]: List of PyVista meshes for the skeleton.
    """
    skel_visual = XMLVisualDataContainer(mjcf_path)
    skel_meshes = []
    for i, mesh in enumerate(skel_visual.meshes):
        # Assign white color to each mesh
        mesh.cell_data["color"] = np.array([[1.0, 1.0, 1.0]]).repeat(mesh.n_cells, 0)
        skel_meshes.append(mesh)
    return skel_meshes


def load_axes(mjcf_path: str) -> List[pv.PolyData]:
    """
    Loads the axes meshes from the MJCF file.

    Args:
        mjcf_path (str): Path to the MJCF file.

    Returns:
        List[pv.PolyData]: List of PyVista meshes for the axes.
    """
    skel_visual = XMLVisualDataContainer(mjcf_path)
    axes_meshes = []
    tip_radius = 0.01
    shaft_radius = 0.005

    for i in range(len(skel_visual.meshes)):
        # Create X, Y, Z axes as arrows
        x_axis_mesh = pv.Arrow(
            (0, 0, 0), (1, 0, 0), tip_radius=tip_radius, shaft_radius=shaft_radius
        )
        x_axis_mesh.cell_data["color"] = (
            np.array([[1, 0, 0]]).repeat(x_axis_mesh.n_cells, 0) * 255
        )
        y_axis_mesh = pv.Arrow(
            (0, 0, 0), (0, 1, 0), tip_radius=tip_radius, shaft_radius=shaft_radius
        )
        y_axis_mesh.cell_data["color"] = (
            np.array([[0, 1, 0]]).repeat(y_axis_mesh.n_cells, 0) * 255
        )
        z_axis_mesh = pv.Arrow(
            (0, 0, 0), (0, 0, 1), tip_radius=tip_radius, shaft_radius=shaft_radius
        )
        z_axis_mesh.cell_data["color"] = (
            np.array([[0, 0, 1]]).repeat(z_axis_mesh.n_cells, 0) * 255
        )
        # Combine the axes into a single mesh
        mesh = x_axis_mesh + y_axis_mesh + z_axis_mesh
        axes_meshes.append(mesh)
    return axes_meshes


def add_skel_meshes(pl: ImguiPlotter, skel_meshes: List[pv.PolyData]) -> List[pv.Actor]:
    """
    Adds skeleton meshes to the ImGui plotter.

    Args:
        pl (ImguiPlotter): The ImGui plotter instance.
        skel_meshes (List[pv.PolyData]): List of skeleton meshes.

    Returns:
        List[pv.Actor]: List of actors representing the skeleton meshes.
    """
    skel_actors = []
    for mesh in skel_meshes:
        actor = pl.add_mesh(
            mesh,
            scalars="color",
            rgb=True,
            smooth_shading=True,
            show_scalar_bar=False,
        )
        skel_actors.append(actor)
    return skel_actors


def add_axes_meshes(pl: ImguiPlotter, axes_meshes: List[pv.PolyData]) -> List[pv.Actor]:
    """
    Adds axes meshes to the ImGui plotter.

    Args:
        pl (ImguiPlotter): The ImGui plotter instance.
        axes_meshes (List[pv.PolyData]): List of axes meshes.

    Returns:
        List[pv.Actor]: List of actors representing the axes meshes.
    """
    axes_actors = []
    for mesh in axes_meshes:
        actor = pl.add_mesh(
            mesh,
            scalars="color",
            rgb=True,
            show_scalar_bar=False,
        )
        axes_actors.append(actor)
    return axes_actors


def set_skel_pose(
    skel_state: SkeletonState,
    skel_actors: List[pv.Actor],
    axes_actors: List[pv.Actor],
    show_axes: bool = True,
):
    """
    Sets the pose of the skeleton based on the skeleton state.

    Args:
        skel_state (SkeletonState): The state of the skeleton.
        skel_actors (List[pv.Actor]): List of skeleton actors.
        axes_actors (List[pv.Actor]): List of axes actors.
        show_axes (bool, optional): Whether to show axes or not. Defaults to True.
    """
    global_translation = skel_state.global_translation
    global_rotation = skel_state.global_rotation
    for i in range(len(skel_actors)):
        # Set visibility of axes actors
        if show_axes:
            axes_actors[i].SetVisibility(True)
        else:
            axes_actors[i].SetVisibility(False)

        # Create a transformation matrix from global rotation and translation
        m = np.eye(4)
        m[:3, :3] = Rotation.from_quat(global_rotation[i]).as_matrix()
        m[:3, 3] = global_translation[i]

        # Apply transformation to both skeleton and axes actors
        skel_actors[i].user_matrix = m
        axes_actors[i].user_matrix = m


def add_3p_meshes(pl: ImguiPlotter) -> List[pv.Actor]:
    """
    Adds three 3D objects to the given PyVista plotter, each consisting of a colored sphere
    with three axes (x, y, z) represented by arrows. The spheres are colored red, green,
    and blue, corresponding to the x, y, and z axes respectively.

    Args:
        pl (ImguiPlotter): The PyVista plotter instance to which the meshes will be added.

    Returns:
        List[pv.Actor]: A list of PyVista actors representing the added 3D objects (sphere and axes).
    """
    actors = []
    for i in range(3):
        # Define sphere color based on iteration index: 0 -> red, 1 -> green, 2 -> blue
        color = {0: [255, 0, 0], 1: [0, 255, 0], 2: [0, 0, 255]}[i]

        # Create a sphere with a radius of 0.1
        sphere = pv.Sphere(radius=0.1)

        # Set the color for the entire surface of the sphere
        sphere.cell_data["color"] = np.array([color] * sphere.n_cells)

        # Define the dimensions of the axes (arrows)
        tip_radius = 0.01
        shaft_radius = 0.005

        # Create an arrow mesh for the x-axis (red) and set its color
        x_axis_mesh = pv.Arrow(
            (0, 0, 0), (1, 0, 0), tip_radius=tip_radius, shaft_radius=shaft_radius
        )
        x_axis_mesh.cell_data["color"] = (
            np.array([[1, 0, 0]]).repeat(x_axis_mesh.n_cells, 0) * 255
        )

        # Create an arrow mesh for the y-axis (green) and set its color
        y_axis_mesh = pv.Arrow(
            (0, 0, 0), (0, 1, 0), tip_radius=tip_radius, shaft_radius=shaft_radius
        )
        y_axis_mesh.cell_data["color"] = (
            np.array([[0, 1, 0]]).repeat(y_axis_mesh.n_cells, 0) * 255
        )

        # Create an arrow mesh for the z-axis (blue) and set its color
        z_axis_mesh = pv.Arrow(
            (0, 0, 0), (0, 0, 1), tip_radius=tip_radius, shaft_radius=shaft_radius
        )
        z_axis_mesh.cell_data["color"] = (
            np.array([[0, 0, 1]]).repeat(z_axis_mesh.n_cells, 0) * 255
        )

        # Combine the sphere and the axis arrows into one mesh
        mesh = sphere + x_axis_mesh + y_axis_mesh + z_axis_mesh

        # Add the combined mesh to the PyVista plotter and store the actor
        actor = pl.add_mesh(
            mesh,
            scalars="color",  # Use the 'color' data for rendering
            rgb=True,  # Colors are provided in RGB format
            smooth_shading=True,  # Enable smooth shading for visual appeal
            show_scalar_bar=False,  # Hide the scalar bar
        )

        # Append the actor to the list of actors
        actors.append(actor)

    # Return the list of actors created
    return actors


def open_copycat_and_pad():
    copycat_path = os.path.join(f"data/amass_copycat_take5_train_medium.pkl")
    with open(copycat_path, "rb") as f:
        d = joblib.load(f)
    # Most sane motion data prep for copycat: nanpad edition
    xyzs = []
    quats = []
    lengths = []
    for k in d.keys():
        xyz = d[k]["trans_orig"]
        quat = d[k]["pose_quat"]
        length = xyz.shape[0]
        xyzs.append(xyz)
        quats.append(quat)
        lengths.append(length)
    max_length = np.max(lengths)
    padded_xyzs = []
    padded_expms = []
    for i in range(len(xyzs)):
        xyz = xyzs[i]
        quat = quats[i]

        euls = Rotation.from_quat(quat[:, 0]).as_euler("xyz")  # "xyz" important because it's extrinsic
        euls[:, 2] = 0
        quat[:, 0] = Rotation.from_euler("xyz", euls).as_quat()
        expm = quat_to_expm(quat)

        padded_xyz = np.pad(xyz, ((0, max_length - lengths[i]), (0, 0)), mode="constant", constant_values=np.nan)
        padded_expm = np.pad(expm, ((0, max_length - lengths[i]), (0, 0), (0, 0)), mode="constant", constant_values=np.nan)
        padded_xyzs.append(padded_xyz)
        padded_expms.append(padded_expm)
    xyzs = np.stack(padded_xyzs, axis=0)
    expms = np.stack(padded_expms, axis=0)
    lengths = np.array(lengths)
    xyzexpms = np.concatenate([xyzs[:, :, None], expms], axis=2)
    vs = np.zeros_like(xyzexpms)
    vs[:, 1:] = xyzexpms[:, 1:] - xyzexpms[:, :-1]
    xyzexpmvs = np.concatenate([xyzexpms, vs], axis=2)
    xyzexpmvs = xyzexpmvs.reshape((xyzexpmvs.shape[0], xyzexpmvs.shape[1], -1))
    zexpmvs = xyzexpmvs[..., 2:]
    return lengths, zexpmvs
