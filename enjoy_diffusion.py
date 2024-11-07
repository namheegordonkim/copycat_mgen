import os
from argparse import ArgumentParser
from threading import Thread

import numpy as np
import pyvista as pv
import torch
from imgui_bundle import immapp
from imgui_bundle._imgui_bundle import imgui, hello_imgui
from pyvista_imgui import ImguiPlotter

from poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree
from torch_nets import UnconditionalEDM
from train_utils import expm_to_quat
from utils import set_skel_pose, load_skel, add_skel_meshes, load_axes, add_axes_meshes, open_copycat_and_pad

MJCF_PATH = "assets/my_smpl_humanoid.xml"
device = torch.device("cuda")


class AppState:
    def __init__(
        self,
        skel_state: SkeletonState,
        skel_actors,
        axes_actors,
        zexpmvs: torch.Tensor,
        lengths: torch.Tensor,
        segment_length: int,
        diffusion_model: UnconditionalEDM,
    ):
        self.axes_actors = axes_actors
        self.diffusion_model = diffusion_model
        self.lengths = lengths
        self.segment_length = segment_length
        self.show_axes = False
        self.skel_actors = skel_actors
        self.skel_state = skel_state
        self.zexpmvs = zexpmvs

        self.denoising_history = []
        self.denoising_history_idx = -1
        self.denoising_render_interval = 100
        self.denoising_t = 100
        self.denoising_yes = False
        self.first_yes = True
        self.gen_frame = 0
        self.playing = False
        self.seq_frame = 0
        self.seq_idx = 0
        self.seq_sigma = 0
        self.zexpmv = self.zexpmvs[self.seq_idx, self.seq_frame]
        self.seed = 0
        self.S_min = 0.2
        self.S_noise = 1
        self.S_churn = 1
        self.num_steps = 50

        self.generated_segment = np.zeros([self.diffusion_model.input_frames, self.zexpmv.shape[-1]])


def do_diffusion_work(app_state: AppState):
    S_noise = app_state.S_noise
    S_churn = app_state.S_churn
    num_steps = app_state.num_steps
    S_min = app_state.S_min
    S_max = np.inf
    sigma_max = app_state.diffusion_model.sigma_max
    sigma_min = app_state.diffusion_model.sigma_min
    rho = 7

    torch.random.manual_seed(app_state.seed)
    torch.cuda.manual_seed(app_state.seed)

    app_state.generated_segment = torch.randn((app_state.diffusion_model.input_frames, app_state.generated_segment.shape[-1]), device=device, dtype=torch.float) * 3
    app_state.denoising_history.append(app_state.generated_segment.clone())

    # Time step discretization.
    step_indices = torch.arange(num_steps, device=device, dtype=torch.float)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = app_state.generated_segment

        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = t_cur + gamma * t_cur
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * torch.randn_like(x_cur)

        # Euler step.
        with torch.no_grad():
            denoised = app_state.diffusion_model.step(app_state.generated_segment[None], t_hat)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 2:
            with torch.no_grad():
                denoised = app_state.diffusion_model.step(x_next, t_next)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        app_state.generated_segment = x_next[0]
        app_state.denoising_history.append(app_state.generated_segment.clone())

    app_state.denoising_yes = False
    app_state.denoising_history = app_state.denoising_history[::-1]


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

        changed1, app_state.seq_idx = imgui.slider_int("Motion Sequence", app_state.seq_idx, 0, app_state.zexpmvs.shape[0] - 1)
        max_frame_for_seq = app_state.lengths[app_state.seq_idx] - 1
        changed2, app_state.seq_frame = imgui.slider_int("Frame in Sequence", app_state.seq_frame, 0, max_frame_for_seq)
        app_state.seq_frame = np.clip(app_state.seq_frame, 0, max_frame_for_seq)
        changed = np.any([changed1, changed2])
        if changed or app_state.first_yes:
            app_state.zexpmv = app_state.zexpmvs[app_state.seq_idx, app_state.seq_frame]
        changed, app_state.show_axes = imgui.checkbox("Show Axes", app_state.show_axes)

        imgui.separator()

        changed, app_state.seed = imgui.slider_int("Seed", app_state.seed, 0, 1000)
        changed, app_state.num_steps = imgui.slider_int("# Denoising Steps", app_state.num_steps, 1, 1000)
        changed, app_state.S_noise = imgui.slider_float("S_noise", app_state.S_noise, 0, 1)
        changed, app_state.S_churn = imgui.slider_float("S_churn", app_state.S_churn, 0, 1)
        changed, app_state.S_min = imgui.slider_float("S_min", app_state.S_min, 0, 80)
        generate_clicked = imgui.button("Generate")
        if generate_clicked and not app_state.denoising_yes:
            app_state.gen_frame = 0
            app_state.denoising_yes = True
            app_state.denoising_t = app_state.num_steps
            app_state.denoising_history = []
            app_state.denoising_history_idx = 0
            worker = Thread(target=do_diffusion_work, args=(app_state,))
            worker.run()

        hist_changed, app_state.denoising_history_idx = imgui.slider_int(
            "Denoising History Index",
            app_state.denoising_history_idx,
            0,
            len(app_state.denoising_history) - 1,
        )
        if (hist_changed or generate_clicked) and len(app_state.denoising_history) > 0:
            app_state.zexpmv = app_state.denoising_history[app_state.denoising_history_idx][app_state.gen_frame].detach().cpu().numpy()
        changed, app_state.gen_frame = imgui.slider_int("Generated Frame", app_state.gen_frame, 0, app_state.diffusion_model.input_frames - 1)
        changed, app_state.playing = imgui.checkbox("Play", app_state.playing)

        imgui.end()

        if len(app_state.denoising_history) > 0:
            if app_state.playing:
                app_state.gen_frame = (app_state.gen_frame + 1) % app_state.diffusion_model.input_frames
            app_state.zexpmv = app_state.denoising_history[app_state.denoising_history_idx][app_state.gen_frame].detach().cpu().numpy()

        with torch.no_grad():
            global_translation = torch.zeros(3)
            global_translation[-1] = app_state.zexpmv[0].item()
            local_rotation = torch.as_tensor(expm_to_quat(app_state.zexpmv[1:73].reshape(-1, 3)))

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
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Data loading
    lengths, zexpmvs = open_copycat_and_pad()
    # zexpmvs = zexpmvs[..., :73]
    segment_length = np.min(lengths)

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
    diffusion_path = args.diffusion_path
    model_d = torch.load(diffusion_path)
    diffusion_model = model_d["model_cls"](*model_d["model_args"], **model_d["model_kwargs"])
    diffusion_model.load_state_dict(model_d["model_state_dict"])
    diffusion_model = diffusion_model.to(device).eval()

    # Run the GUI
    app_state = AppState(sk_state, sk_actors, ax_actors, zexpmvs, lengths, segment_length, diffusion_model)
    setup_and_run_gui(pl, app_state)

    print(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--diffusion_path", type=str, required=True)
    args = parser.parse_args()

    main()
