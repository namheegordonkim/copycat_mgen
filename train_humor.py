import os
from argparse import ArgumentParser
from functools import reduce

import numpy as np
import torch

import my_logging
from tensorboardX import SummaryWriter
from torch.optim import RAdam
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch_nets import ConditionalVAEWithPrior
from train_utils import ThroughDataset
from utils import open_copycat_and_pad


def get_rollout(model, segment, teacher_forcing_ratio: float = 0.0, noise_std: float = 0.0):
    model.eval()
    with torch.no_grad():
        xins = []
        wins = [segment[:, [0]]]
        xouts = []
        xgts = []
        for i in range(1, segment.shape[1] - 1):
            xin = segment[:, [i]]
            win = wins[-1]
            yes = np.random.rand() < teacher_forcing_ratio
            win[yes] = segment[:, [i - 1]][yes]
            noise = torch.randn_like(win) * noise_std
            win = model.rms_cond.unnormalize(model.rms_cond.normalize(win) + noise)
            wins[-1] = win
            xgt = segment[:, [i]]

            z, decoded, mu, log_var, prior_mu, prior_log_var = model.forward(xin, win)
            xout = decoded

            xins.append(xin)
            wins.append(xout)
            xouts.append(xout)
            xgts.append(xgt)

        xins = torch.cat(xins, dim=1)
        wins = torch.cat(wins[:-1], dim=1)
        xouts = torch.cat(xouts, dim=1)
        xgts = torch.cat(xgts, dim=1)
    model.train()
    return xins, wins, xouts, xgts


def main():
    device = torch.device("cuda")

    logdir = f"log/{args.out_name}"
    outdir = f"out/{args.out_name}"
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    logger = my_logging.get_logger(f"{args.out_name}", logdir)
    logger.info(f"Starting")
    writer = SummaryWriter(logdir)
    writer.add_text("args", str(args))

    # Data loading
    lengths, zexpmvs = open_copycat_and_pad()

    init_segment_length = args.init_segment_length
    max_segment_length = args.max_segment_length

    all_i = np.arange(zexpmvs.shape[0])
    all_t = np.arange(zexpmvs.shape[1])
    all_it = np.stack(np.meshgrid(all_i, all_t), axis=-1).transpose(1, 0, 2).reshape(-1, 2)
    # based on sequence length, filter out (i,t) pairs whose t values exceed valid length
    good_it = all_it[all_it[:, 1] < lengths[all_it[:, 0]] - max_segment_length + 1]

    # Setting up 80-20 train/valid split
    train_idxs = np.random.choice(good_it.shape[0], int(0.8 * good_it.shape[0]), replace=False)
    valid_idxs = np.setdiff1d(np.arange(good_it.shape[0]), train_idxs)
    train_it = good_it[train_idxs]
    valid_it = good_it[valid_idxs]

    it_dataset = ThroughDataset(train_it)
    it_dataloader = DataLoader(it_dataset, batch_size=128, shuffle=False)
    it_dataloader_iter = iter(it_dataloader)

    # For consistency, we will use the same samples for training and validation error calculation
    batch_size = args.batch_size
    seen_it = train_it[:batch_size]
    unseen_it = valid_it[:batch_size]

    idxs = np.arange(max_segment_length)
    seen_segment_idxs = seen_it[:, 1][:, None] + idxs
    seen_segments = zexpmvs[seen_it[:, 0][:, None], seen_segment_idxs]
    seen_segments = torch.as_tensor(seen_segments, dtype=torch.float, device=device)
    unseen_segment_idxs = unseen_it[:, 1][:, None] + idxs
    unseen_segments = zexpmvs[unseen_it[:, 0][:, None], unseen_segment_idxs]
    unseen_segments = torch.as_tensor(unseen_segments, dtype=torch.float, device=device)

    # Training configuration
    n_total_epochs = args.n_total_epochs
    pbar = tqdm(total=n_total_epochs)
    epochs_elapsed = 0
    steps_elapsed = 0
    save_every = n_total_epochs // 10
    eval_every = save_every // 10
    n_warmup_epochs = save_every // 10

    model = ConditionalVAEWithPrior(
        input_size=seen_segments.shape[-1],
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        cond_size=seen_segments.shape[-1],
    )
    model = model.to(device)
    model.rms_input.update(seen_segments.reshape(-1, seen_segments.shape[-1]))
    model.rms_cond.update(seen_segments.reshape(-1, seen_segments.shape[-1]))
    optimizer = RAdam(model.parameters(), lr=args.peak_lr)
    model = torch.compile(model)
    orig_mod = model._orig_mod

    while epochs_elapsed <= n_total_epochs:
        if args.schedule_yes:
            segment_length = np.minimum(init_segment_length + epochs_elapsed // save_every, max_segment_length)
        else:
            segment_length = max_segment_length

        if epochs_elapsed % save_every == 0 or epochs_elapsed >= n_total_epochs:
            pauli_root = os.path.abspath(os.path.join(os.getcwd(), "src"))  # NOTE: assume that all scripts are run from the parent directory of src.
            model_d = {
                "model_cls": orig_mod.__class__,
                "model_args": orig_mod.args,
                "model_kwargs": orig_mod.kwargs,
                "model_state_dict": orig_mod.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epochs_elapsed": epochs_elapsed,
                "args": args,
            }
            save_path = f"{outdir}/humor_{epochs_elapsed:06d}.pkl"
            torch.save(model_d, save_path)
            logger.info(f"Saved to {save_path}")

        if epochs_elapsed >= n_total_epochs:
            break

        # Evaluation
        if epochs_elapsed % eval_every == 0:
            model.eval()
            with torch.no_grad():
                for name, segments in [
                    ("seen", seen_segments),
                    ("unseen", unseen_segments),
                ]:
                    segments = segments.to(device)
                    xins, wins, xouts, xgts = get_rollout(model, segments, args.tf_ratio, 0)

                    idxs = torch.arange(xins.shape[1], device="cuda")
                    batch_idxs = torch.split(idxs, 1)
                    mse_losses = []
                    kld_losses = []
                    for batch_i in batch_idxs:
                        mse_loss = torch.nn.functional.mse_loss(xouts[:, batch_i], xgts[:, batch_i])
                        z, decoded, mu, log_var, prior_mu, prior_log_var = model.forward(xins[:, batch_i], wins[:, batch_i])
                        left = torch.distributions.Normal(mu, log_var.exp().sqrt())
                        right = torch.distributions.Normal(prior_mu, prior_log_var.exp().sqrt())
                        kld_loss = torch.distributions.kl.kl_divergence(left, right).mean()
                        mse_losses.append(mse_loss)
                        kld_losses.append(kld_loss)
                    mse_loss = torch.stack(mse_losses).mean()
                    kld_loss = torch.stack(kld_losses).mean()
                    loss = mse_loss + args.kld_weight * kld_loss
                    mse_kld_ratio = np.maximum(mse_loss.item() / kld_loss.item(), kld_loss.item() / mse_loss.item())
                    for epoch_or_step, value in zip(["epochs", "steps"], [epochs_elapsed, steps_elapsed]):
                        writer.add_scalar(f"{name}/MSELoss/{epoch_or_step}", mse_loss.item(), value)
                        writer.add_scalar(f"{name}/KLDLoss/{epoch_or_step}", kld_loss.item(), value)
                        writer.add_scalar(f"{name}/MSEKLDRaio/{epoch_or_step}", mse_kld_ratio, value)
                        writer.add_scalar(f"{name}/loss/{epoch_or_step}", loss.item(), value)
                    logger.info(f"Epoch {epochs_elapsed}: {name} MSELoss: {mse_loss.item():.2e} KLDLoss: {kld_loss.item():.2e} MSEKLDRatio: {mse_kld_ratio.item():.2e}")
            model.train()

        for (x_it,) in it_dataloader:
            # x_it, = next(it_dataloader_iter)
            idxs = torch.arange(segment_length)
            x_idxs = x_it[:, 1][:, None] + idxs
            x_segments = zexpmvs[x_it[:, 0][:, None], x_idxs]
            x_segments = torch.as_tensor(x_segments, dtype=torch.float, device=device)
            xins, wins, xouts, xgts = get_rollout(model, x_segments, args.tf_ratio, args.noise_std)

            xins = xins.reshape((-1, 1, xins.shape[-1]))
            wins = wins.reshape((-1, 1, wins.shape[-1]))
            xgts = xgts.reshape((-1, 1, xgts.shape[-1]))
            # Then supervised learning from input
            dataset = ThroughDataset(xins, wins, xgts)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            min_lr = 0
            if args.lr_decay_yes:
                if epochs_elapsed % save_every < n_warmup_epochs:
                    lr = min_lr + (args.peak_lr - min_lr) * np.clip((epochs_elapsed % save_every) / n_warmup_epochs, 0, 1)
                else:
                    # cosine decay
                    lr = min_lr + (args.peak_lr - min_lr) * (1 + np.cos(np.pi * (epochs_elapsed % save_every - n_warmup_epochs) / (save_every - n_warmup_epochs))) / 2
            else:
                lr = args.peak_lr
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            writer.add_scalar("train/lr", lr, epochs_elapsed)

            for x_in, w_in, x_gt in dataloader:
                x_in = x_in.to(device)
                w_in = w_in.to(device)
                x_gt = x_gt.to(device)

                z, x_hat, mu, log_var, prior_mu, prior_log_var = model.forward(x_in, w_in)

                my_dist = torch.distributions.Normal(mu, log_var.exp().sqrt())
                prior_dist = torch.distributions.Normal(prior_mu, prior_log_var.exp().sqrt())

                recon_loss = torch.nn.functional.mse_loss(x_hat, x_gt)
                kld_loss = torch.distributions.kl.kl_divergence(my_dist, prior_dist).mean()
                loss = recon_loss + args.kld_weight * kld_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                steps_elapsed += 1

        epochs_elapsed += 1
        pbar.update(1)
        pbar.set_postfix({"loss": loss.item(), "epochs": epochs_elapsed})

    pbar.close()

    logger.info(f"Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--kld_weight", type=float, default=4e-4)
    parser.add_argument("--n_total_epochs", type=int, default=int(1e2))
    parser.add_argument("--continue_yes", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--peak_lr", type=float, default=3e-4)
    parser.add_argument("--lr_decay_yes", action="store_true")
    parser.add_argument("--latent_size", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--init_segment_length", type=int, default=2)
    parser.add_argument("--max_segment_length", type=int, default=11)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--schedule_yes", action="store_true")
    parser.add_argument("--tf_ratio", type=float, default=0.1)
    args = parser.parse_args()

    main()
