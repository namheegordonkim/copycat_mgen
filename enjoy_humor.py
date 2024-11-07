from argparse import ArgumentParser
from imgui_bundle import immapp
from imgui_bundle._imgui_bundle import imgui, hello_imgui
from poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree
from pyvista_imgui import ImguiPlotter
from torch_nets import ConditionalVAEWithPrior
from train_utils import expm_to_quat_torch
from utils import set_skel_pose, load_skel, add_skel_meshes, load_axes, add_axes_meshes, open_copycat_and_pad
import numpy as np
import pyvista as pv
import torch

MJCF_PATH = "assets/my_smpl_humanoid.xml"
device = torch.device("cuda")


class AppState:
    def __init__(
        self,
        skel_state: SkeletonState,
        skel_actors,
        axes_actors,
        zexpmvs: torch.Tensor,
        humor: ConditionalVAEWithPrior,
    ):
        self.skel_state = skel_state
        self.skel_actors = skel_actors
        self.axes_actors = axes_actors
        self.zexpmvs = zexpmvs
        self.humor = humor

        self.first_yes = True
        self.latents = torch.zeros(humor.latent_size, dtype=torch.float, device=device)
        self.playing = False
        self.pose_idx = 0
        self.show_axes = False
        self.w = None
        self.x = None


def setup_and_run_gui(pl: ImguiPlotter, app_state: AppState):
    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "Viewer"
    runner_params.app_window_params.window_geometry.size = (1280, 720)
    runner_params.fps_idling.enable_idling = False

    def gui():
        hello_imgui.apply_theme(hello_imgui.ImGuiTheme_.imgui_colors_dark)

        viewport_size = imgui.get_window_viewport().size

        # PyVista portion
        imgui.set_next_window_size(imgui.ImVec2(viewport_size.x // 2, viewport_size.y))
        imgui.set_next_window_pos(imgui.ImVec2(viewport_size.x // 2, 0))
        imgui.set_next_window_bg_alpha(1.0)
        imgui.begin(
            "ImguiPlotter",
            flags=imgui.WindowFlags_.no_bring_to_front_on_focus | imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move,
        )
        # render the plotter's contents here
        pl.render_imgui()
        imgui.end()

        # GUI portion
        imgui.set_next_window_size(imgui.ImVec2(viewport_size.x // 2, viewport_size.y))
        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        imgui.set_next_window_bg_alpha(1.0)
        imgui.begin(
            "Controls",
            flags=imgui.WindowFlags_.no_bring_to_front_on_focus | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move,
        )

        changed, app_state.pose_idx = imgui.slider_int("Pose Index", app_state.pose_idx, 0, app_state.zexpmvs.shape[0] - 2)
        # If pose idx changes, then use the current pose to calculate latents
        if changed or app_state.first_yes:
            zexpmv0 = app_state.zexpmvs[app_state.pose_idx]
            zexpmv1 = app_state.zexpmvs[app_state.pose_idx + 1]
            with torch.no_grad():
                app_state.x = torch.tensor(zexpmv0, dtype=torch.float, device=device)
                app_state.w = torch.tensor(zexpmv1, dtype=torch.float, device=device)
                z, decoded, mu, log_var, prior_mu, prior_log_var = app_state.humor.forward(app_state.x[None, None], app_state.w[None, None])
                app_state.latents[:] = mu[0, 0]

        changed, app_state.playing = imgui.checkbox("Play", app_state.playing)
        changed, app_state.show_axes = imgui.checkbox("Show Axes", app_state.show_axes)

        changeds = np.zeros(app_state.humor.latent_size, dtype=bool)
        for i in range(app_state.humor.latent_size):
            changeds[i], app_state.latents[i] = imgui.slider_float(f"Latent {i}", app_state.latents[i], -3, 3)

        imgui.end()

        with torch.no_grad():
            zexpm = app_state.humor.decode(app_state.latents[None, None], app_state.w[None, None])[0, 0]
            global_translation = torch.zeros(3)
            global_translation[-1] = zexpm[0]
            local_rotation = torch.as_tensor(expm_to_quat_torch(zexpm[1:73].reshape(-1, 3)))

        if app_state.playing:
            app_state.w = app_state.x * 1
            app_state.x = zexpm * 1
            z, decoded, mu, log_var, prior_mu, prior_log_var = app_state.humor.forward(app_state.x[None, None], app_state.w[None, None])
            app_state.latents[:] = mu[0, 0]

        # Set the character pose
        app_state.skel_state = SkeletonState.from_rotation_and_root_translation(
            app_state.skel_state.skeleton_tree,
            local_rotation.cpu().detach(),
            global_translation.cpu().detach(),
            is_local=True,
        )

        set_skel_pose(
            app_state.skel_state,
            app_state.skel_actors,
            app_state.axes_actors,
            app_state.show_axes,
        )

        app_state.first_yes = False

    runner_params.callbacks.show_gui = gui
    runner_params.imgui_window_params.default_imgui_window_type = hello_imgui.DefaultImGuiWindowType.no_default_window
    immapp.run(runner_params=runner_params)


def main():
    lengths, zexpmvs = open_copycat_and_pad()
    # Remove NaNs
    zexpmvs = zexpmvs[~np.isnan(zexpmvs).any(axis=(-1))]

    pl = ImguiPlotter()
    pl.enable_shadows()
    pl.add_axes()
    pl.camera.position = (5, -5, 3)
    pl.camera.focal_point = (0, 0, 1)
    pl.camera.up = (0, 0, 1)

    # Initialize meshes
    floor = pv.Plane(i_size=10, j_size=10)
    skels = load_skel(MJCF_PATH)
    axes = load_axes(MJCF_PATH)

    # Register meshes, get actors for object manipulation
    pl.add_mesh(floor, show_edges=True, pbr=True, roughness=0.24, metallic=0.1)
    sk_actors = add_skel_meshes(pl, skels)
    ax_actors = add_axes_meshes(pl, axes)

    # Set character pose to default
    # Center the character root at the origin
    root_translation = torch.zeros(3)
    # Set global rotation as unit quaternion
    body_part_global_rotation = torch.zeros(24, 4)
    body_part_global_rotation[..., -1] = 1

    # `poselib` handles the forward kinematics
    sk_tree = SkeletonTree.from_mjcf(MJCF_PATH)
    sk_state = SkeletonState.from_rotation_and_root_translation(sk_tree, body_part_global_rotation, root_translation, is_local=False)
    set_skel_pose(sk_state, sk_actors, ax_actors, show_axes=False)

    # Load the model
    humor_path = args.humor_path
    model_d = torch.load(humor_path)
    humor = model_d["model_cls"](*model_d["model_args"], **model_d["model_kwargs"])
    humor.load_state_dict(model_d["model_state_dict"])
    humor = humor.to(device)

    # Run the GUI
    app_state = AppState(sk_state, sk_actors, ax_actors, zexpmvs, humor)
    setup_and_run_gui(pl, app_state)

    print(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--humor_path", type=str, required=True)
    args = parser.parse_args()

    main()
