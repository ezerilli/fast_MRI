import datetime
from pathlib import Path
from argparse import ArgumentParser
from typing import List
from self_supervised import MriSelfSupervised
from ssl_transform import SslTransform
import pytorch_lightning as pl

from fastmri.pl_modules import FastMriDataModule

import fastmri
from fastmri.data import subsample
import numpy as np
import torch.optim
from torch.utils.data import DataLoader


def handle_args():
    parser = ArgumentParser()

    #num_gpus = 0
    #backend = "ddp_cpu"
    #batch_size = 1 if backend == "ddp" else num_gpus
    #batch_size = 2

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        dest='mode',
        help="Operation mode"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        type=str,
        dest='device',
        help="Device type",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        dest='data_path',
        help="Path to data",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        required=True,
        dest='checkpoint_path',
        help="When train, dir path for saving model checkpoints; when test, either director (from which to load newest"
             " checkpoint) or specific checkpoint file to load",
    )
    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced"),
        default="random",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerator",
        dest='accelerator',
        default='ddp',
        help="What distributed version to use",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration rates to use for masks",
    )
    parser.add_argument("--non_deterministic", action='store_false', default=True, dest='deterministic')
    parser.add_argument("--replace_sampler_ddp", action='store_true', default=False, dest='replace_sampler_ddp',
                        help="Replace sampler ddp")
    parser.add_argument("--seed", default=42, dest='seed', help="Seed for all the random generators")
    parser.add_argument("--num_gpus", default=1, help="The number of available GPUs (when device is 'cuda'")

    return parser.parse_args()


def get_sorted_checkpoint_files(checkpoint_dir: Path) -> List[Path]:
    files = list(checkpoint_dir.glob('*.pt'))
    files.sort()
    return files


def save_checkpoint(model: torch.nn.Module, checkpoint_dir: Path, limit=None):
    filename = 'ssl_sd_checkpoint_{}.pt'.format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir()
        torch.save(model.state_dict(), checkpoint_dir.joinpath(filename))
    else:
        torch.save(model.state_dict(), checkpoint_dir.joinpath(filename))
        files = get_sorted_checkpoint_files(checkpoint_dir)
        if limit and len(files) > limit:
            files[0].unlink()


def load_from_checkpoint(model: torch.nn.Module, checkpoint_dir: Path, specific_file: str = None, set_eval=True):
    if specific_file is None:
        files = get_sorted_checkpoint_files(checkpoint_dir)
        file_path = files[-1]
    else:
        file_path = checkpoint_dir.joinpath(specific_file)
    model.load_state_dict(torch.load(file_path))
    if set_eval:
        model.eval()


def calc_ssl_loss(u, v):
    abs_u_minus_v = torch.abs(u - v)
    abs_u = torch.abs(u)
    term_1 = torch.pow(abs_u_minus_v, 2) / torch.pow(abs_u, 2)
    term_2 = abs_u_minus_v / abs_u
    return term_1 + term_2


def choose_loss_split(volume, ratio=0.5):
    # TODO: come back and implement overlap
    arange = np.arange(volume.shape[0])
    theta_indices = np.random.Generator.choice(arange, size=int(volume.shape[0] * ratio), replace=False)
    lambda_indices = arange[np.isin(arange, theta_indices, invert=True)]
    volume_theta_view = volume[theta_indices]
    volume_lambda_view = volume[lambda_indices]
    return volume_theta_view, volume_lambda_view


def run_training_for_volume(volume, model: torch.nn.Module, optimizer):
    volume_theta_view, volume_lambda_view = choose_loss_split(volume)
    prediction = model(volume_theta_view)
    loss = calc_ssl_loss(u=volume_lambda_view, v=prediction)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def run_training(model: torch.nn.Module, checkpoint_dir: Path, dataloader: DataLoader, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for e in range(epochs):
        for volume in dataloader:
            loss = run_training_for_volume(volume, model, optimizer)
        save_checkpoint(model, checkpoint_dir)
        print(f"loss: {loss:>7f}  [{e:>5d}/{epochs:>5d}]")


def run_pretrained_inference(checkpoint_source: Path):
    # TODO: source may be directory (newest file) or actual file to load
    # TODO: implement
    raise RuntimeError("Testing mode not currently supported")


def main():
    args = handle_args()
    pl.seed_everything(args.seed)

    # creates k-space mask for transforming
    mask = subsample.create_mask_for_mask_type(args.mask_type, args.center_fractions, args.accelerations)
    train_transform = SslTransform(mask_func=mask, use_seed=False)
    val_transform = SslTransform(mask_func=mask)
    test_transform = SslTransform()

    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge='singlecoil',
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split='test',
        # TODO: this in particular might need to be changed
        test_path=None,
        sample_rate=None,
        batch_size=1,
        num_workers=4,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint_path = args.checkpoint_path
    if args.mode == "train":
        if checkpoint_path.exists() and not checkpoint_path.is_dir():
            raise RuntimeError("Existing, non-directory path {} given for checkpoint directory".format(checkpoint_path))
        model = MriSelfSupervised()
        model.to(device)
        run_training(model=model, checkpoint_dir=checkpoint_path, dataloader=data_module.train_dataloader())
    elif args.mode == "test":
        if not checkpoint_path.exists():
            raise RuntimeError("Non-existing checkpoint file/directory path {}".format(checkpoint_path))
        run_pretrained_inference(checkpoint_source=checkpoint_path)
    else:
        raise RuntimeError("Unsupported mode '{}'".format(args.mode))


if __name__ == "__main__":
    main()
