"""Clustering hits into readout cells using a regular grid."""

import argparse
import multiprocessing
import time

import numpy as np
import numpy.typing as npt

from . import core as showerdata
from . import detector
from .core import Showers

__all__ = ["cluster"]


def _version_ge(v1: str, v2: str) -> bool:
    """Check if version string v1 is greater than or equal to v2."""
    v1 = v1.strip().lstrip("v")
    v2 = v2.strip().lstrip("v")
    parts1 = tuple(int(p) for p in v1.split(".") if p.isdigit())
    parts2 = tuple(int(p) for p in v2.split(".") if p.isdigit())

    # Pad the shorter version with zeros
    length = max(len(parts1), len(parts2))
    parts1 += (0,) * (length - len(parts1))
    parts2 += (0,) * (length - len(parts2))

    return parts1 >= parts2


def _cluster_shower_part(
    pos: npt.NDArray[np.float32],
    e: npt.NDArray[np.float32],
    cell_size: float,
    shift: npt.NDArray[np.float32],
    t: npt.NDArray[np.float32] | None = None,
) -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32] | None
]:
    """Cluster a part of a shower (ECAL or HCAL)."""
    if len(pos) == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            None if t is None else np.empty((0,), dtype=np.float32),
        )
    pos = pos.copy()
    pos[:, :2] += shift
    pos[:, :2] /= cell_size
    pos_idx = np.empty_like(pos, dtype=np.int32)
    pos_idx = np.floor(pos, out=pos_idx, casting="unsafe")
    pos_idx = pos_idx[:, [2, 0, 1]]  # z, x, y

    # Prior to numpy 2.3 np.unique did not have a 'sorted' argument
    if _version_ge(np.__version__, "2.3"):
        unique_idx, inverse_idx = np.unique(
            pos_idx, axis=0, return_inverse=True, sorted=True
        )
    else:
        unique_idx, inverse_idx = np.unique(pos_idx, axis=0, return_inverse=True)
    unique_idx = unique_idx[:, [1, 2, 0]]  # x, y, z
    e_clustered = np.zeros((len(unique_idx),), dtype=np.float32)
    np.add.at(e_clustered, inverse_idx, e)
    if t is not None:
        t_clustered = np.full((len(unique_idx),), np.inf, dtype=np.float32)
        np.minimum.at(t_clustered, inverse_idx, t)
    else:
        t_clustered = None
    pos_clustered = unique_idx.astype(np.float32)
    pos_clustered[:, :2] += 1 / 2
    pos_clustered[:, :2] *= cell_size
    pos_clustered[:, :2] -= shift

    return pos_clustered, e_clustered, t_clustered


def _calc_random_shift(cell_size: float, random_shift: bool) -> npt.NDArray[np.float32]:
    """Calculate a random shift for the grid."""
    if random_shift:
        return (np.random.rand(2).astype(np.float32) * cell_size) - cell_size / 2
    else:
        return np.array([0.0, 0.0], dtype=np.float32)


def _process_shower(
    shower: Showers,
    random_shift: bool,
    detector_config: detector.DetectorGeometry,
) -> Showers:
    """Process a single shower by clustering its hits."""
    pos = shower.points[0, :, :3]
    e = shower.points[0, :, 3]
    t = shower.points[0, :, 4] if shower.points.shape[2] > 4 else None
    mask = e > 0
    pos = pos[mask]
    e = e[mask]
    if t is not None:
        t = t[mask]

    ecal_mask = pos[:, 2] < detector_config.num_layers_ecal - 0.5
    hcal_mask = pos[:, 2] >= detector_config.num_layers_ecal - 0.5
    pos_clustered_ecal, e_clustered_ecal, t_clustered_ecal = _cluster_shower_part(
        pos=pos[ecal_mask],
        e=e[ecal_mask],
        cell_size=detector_config.ecal_cell_size,
        shift=_calc_random_shift(detector_config.ecal_cell_size, random_shift),
        t=t[ecal_mask] if t is not None else None,
    )
    pos_clustered_hcal, e_clustered_hcal, t_clustered_hcal = _cluster_shower_part(
        pos=pos[hcal_mask],
        e=e[hcal_mask],
        cell_size=detector_config.hcal_cell_size,
        shift=_calc_random_shift(detector_config.hcal_cell_size, random_shift),
        t=t[hcal_mask] if t is not None else None,
    )

    pos_clustered = np.concatenate((pos_clustered_ecal, pos_clustered_hcal), axis=0)
    e_clustered = np.concatenate((e_clustered_ecal, e_clustered_hcal), axis=0)
    points_clustered = np.zeros((1, len(mask), 4 if t is None else 5), dtype=np.float32)
    points_clustered[0, : len(pos_clustered), :3] = pos_clustered
    points_clustered[0, : len(e_clustered), 3] = e_clustered
    if t is not None:
        if t_clustered_ecal is None or t_clustered_hcal is None:
            raise RuntimeError("Time information missing in one of the shower parts.")
        t_clustered = np.concatenate((t_clustered_ecal, t_clustered_hcal), axis=0)
        points_clustered[0, : len(t_clustered), 4] = t_clustered

    return Showers(
        points=points_clustered,
        energies=shower.energies,
        pdg=shower.pdg,
        directions=shower.directions,
        shower_ids=shower.shower_ids,
    )


def cluster(
    showers: Showers,
    random_shift: bool = True,
    detector_config: detector.DetectorGeometry = detector.get_ILD_geometry(),
    processes: int = 1,
) -> Showers:
    """Cluster hits into readout cells using a regular grid.

    Args:
        showers: Showers to cluster.
        random_shift: Whether to apply a random shift to the grid (default: True).
        detector_config: Simplified detector description (default: ILD).
        processes: Number of parallel processes to use (default: 1, i.e. no parallelism).

    Returns:
        Clustered showers.
    """
    if processes > 1:
        with multiprocessing.Pool(processes) as pool:
            processed_showers = pool.starmap(
                _process_shower,
                [(shower, random_shift, detector_config) for shower in showers],
            )
    else:
        processed_showers = [
            _process_shower(shower, random_shift, detector_config) for shower in showers
        ]

    return showerdata.concatenate(processed_showers)


def add_parser_arguments(parser: "argparse.ArgumentParser") -> None:
    """Add arguments for the clustering module to an argparse parser."""
    parser.add_argument(
        "input", type=str, help="Input file containing showers to be clustered."
    )
    parser.add_argument(
        "output", type=str, help="Output file to save the clustered showers."
    )
    parser.add_argument(
        "--no-random-shift",
        action="store_true",
        help="Disable random shift of the grid.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Number of showers to process in each batch.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of parallel processes to use (default: 1, i.e. no parallelism).",
    )


def initialize_parser() -> argparse.ArgumentParser:
    """Initialize an argparse parser for the clustering module."""
    parser = argparse.ArgumentParser(
        description="Cluster hits into readout cells using a regular grid."
    )
    add_parser_arguments(parser)
    return parser


def main(args: argparse.Namespace) -> None:
    """Main function to run clustering from command line arguments."""
    start_time = time.time()
    shape = showerdata.get_file_shape(args.input)
    with (
        showerdata.ShowerDataFile(args.input, "r") as in_file,
        showerdata.ShowerDataFile(args.output, "w", shape=shape) as out_file,
    ):
        out_file.attrs["clustered_for"] = "ILD"
        for start in range(0, shape[0], args.batch_size):
            end = min(start + args.batch_size, shape[0])
            showers = in_file[start:end]
            clustered_showers = cluster(
                showers,
                random_shift=not args.no_random_shift,
                detector_config=detector.get_ILD_geometry(),
                processes=args.processes,
            )
            out_file[start:end] = clustered_showers
            print(
                f"[{time.time() - start_time:8.2f}s] Processed showers {start} to {end}."
            )


if __name__ == "__main__":
    main(initialize_parser().parse_args())
