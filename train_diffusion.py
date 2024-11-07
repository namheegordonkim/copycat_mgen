import os
from argparse import ArgumentParser

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim import RAdam, AdamW
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

import my_logging
from torch_nets import UnconditionalEDM
from train_utils import ThroughDataset
from utils import open_copycat_and_pad


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

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
    segment_length = args.segment_length
    yes = lengths >= segment_length
    lengths = lengths[yes]
    zexpmvs = zexpmvs[yes]

    all_i = np.arange(zexpmvs.shape[0])
    all_t = np.arange(zexpmvs.shape[1])
    all_it = np.stack(np.meshgrid(all_i, all_t), axis=-1).transpose(1, 0, 2).reshape(-1, 2)
    # based on sequence length, filter out (i,t) pairs whose t values exceed valid length
    good_it = all_it[all_it[:, 1] < lengths[all_it[:, 0]] - segment_length + 1]

    train_idxs = np.random.choice(good_it.shape[0], int(0.95 * good_it.shape[0]), replace=False)
    valid_idxs = np.setdiff1d(np.arange(good_it.shape[0]), train_idxs)
    train_it = good_it[train_idxs]
    valid_it = good_it[valid_idxs]

    it_dataset = ThroughDataset(train_it)
    it_dataloader = DataLoader(
        it_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        # sampler=RandomSampler(it_dataset, replacement=True, num_samples=args.batch_size)
    )
    it_dataloader_iter = iter(it_dataloader)

    # For consistency, we will use the same samples for training and validation error calculation
    batch_size = args.batch_size
    seen_it = train_it[:batch_size]
    unseen_it = valid_it[:batch_size]

    idxs = np.arange(segment_length)
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

    model = UnconditionalEDM(
        input_size=seen_segments.shape[-1],
        input_frames=segment_length,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    model = model.to(device)
    model.rms_input.update(seen_segments.reshape(-1, seen_segments.shape[-1]))
    optimizer = AdamW(model.parameters(), lr=args.peak_lr)
    model = torch.compile(model)
    orig_mod = model._orig_mod

    while epochs_elapsed <= n_total_epochs:

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
            save_path = f"{outdir}/edm_{epochs_elapsed:06d}.pkl"
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
                    eps_out, eps_tar, lammy = model.forward(segments)
                    losses = lammy * torch.mean((eps_out - eps_tar).abs() ** 2, dim=(-1, -2))
                    loss = torch.mean(losses)

                    for epoch_or_step, value in zip(["epochs", "steps"], [epochs_elapsed, steps_elapsed]):
                        writer.add_scalar(f"{name}/loss/{epoch_or_step}", loss.item(), value)
                        writer.add_scalar(f"{name}/losses[0]/{epoch_or_step}", losses[0].item(), value)
                        writer.add_scalar(f"{name}/losses[1]/{epoch_or_step}", losses[1].item(), value)

                    logger.info(f"Epoch {epochs_elapsed}: {name} loss: {loss.item():.2e}")
            model.train()

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

        for (x_it,) in it_dataloader:
            idxs = torch.arange(segment_length)
            x_idxs = x_it[:, 1][:, None] + idxs
            x_segments = zexpmvs[x_it[:, 0][:, None], x_idxs]
            x_segments = torch.as_tensor(x_segments, dtype=torch.float, device=device)
            eps_out, eps_tar, lammy = model.forward(x_segments)
            losses = lammy * torch.mean((eps_out - eps_tar).abs() ** 2, dim=(-1, -2))
            loss = torch.mean(losses)

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
    parser.add_argument("--n_total_epochs", type=int, default=int(1e2))
    parser.add_argument("--continue_yes", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--peak_lr", type=float, default=5e-5)
    parser.add_argument("--lr_decay_yes", action="store_true")
    parser.add_argument("--latent_size", type=int, default=1024)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--schedule_yes", action="store_true")
    parser.add_argument("--segment_length", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    args = parser.parse_args()

    main()
