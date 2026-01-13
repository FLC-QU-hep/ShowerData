"""\
Functionality for filtering shower data based on spatial and energy criteria.
Can be used to clean up data for classification trainings or other applications.
"""

import argparse
import os
import time
import warnings

import numpy as np

from . import core, detector


def filter_showers(
    shower_data: core.Showers,
    radius: float = float("inf"),
    ecal_threshold: float = 0.0,
    hcal_threshold: float = 0.0,
    num_layers_ecal: int = 30,
) -> core.Showers:
    """\
    Filter hists in the shower data based on the specified criteria.

    Args:
        shower_data: The shower data to filter.
        radius: Radius (in millimeters) for the cylindrical cut filter.
        ecal_threshold: Energy threshold (in GeV) for the ECAL hit filter.
        hcal_threshold: Energy threshold (in GeV) for the HCAL hit filter.
        num_layers_ecal: Number of layers in the ECAL detector.

    Returns:
        The filtered shower data.
    """
    points = shower_data.points
    mask = (points[:, :, 0] ** 2 + points[:, :, 1] ** 2 < radius**2) & (
        (
            (points[:, :, 2] < num_layers_ecal - 0.5)
            & (points[:, :, 3] >= ecal_threshold)
        )
        | (
            (points[:, :, 2] >= num_layers_ecal - 0.5)
            & (points[:, :, 3] >= hcal_threshold)
        )
    )

    # loop avoids excessive memory usage and compute expensive sorting operations
    filtered_points = np.zeros_like(points)
    for i in range(points.shape[0]):
        valid_points = points[i, mask[i]]
        filtered_points[i, : valid_points.shape[0]] = valid_points

    return core.Showers(
        points=filtered_points,
        energies=shower_data.energies.copy(),
        pdg=shower_data.pdg.copy(),
        directions=shower_data.directions.copy(),
        shower_ids=shower_data.shower_ids.copy(),
    )


def filter_file(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    radius: float = float("inf"),
    ecal_threshold: float = 0.0,
    hcal_threshold: float = 0.0,
    num_layers_ecal: int = 30,
    overwrite: bool = False,
    batch_size: int = 1000,
    verbose: bool = False,
) -> None:
    """\
    Filter showers in a file and save the filtered showers to a new file.

    Args:
        input_path: Path to the input file containing shower data.
        output_path: Path to the output file to save the filtered shower data.
        radius: Radius (in millimeters) for the cylindrical cut filter.
        ecal_threshold: Energy threshold (in GeV) for the ECAL hit filter.
        hcal_threshold: Energy threshold (in GeV) for the HCAL hit filter.
        num_layers_ecal: Number of layers in the ECAL detector.
        overwrite: Whether to overwrite the output file if it already exists.
        batch_size: Number of showers to process in each batch.
        verbose: Whether to print progress information.
    """
    if os.path.exists(output_path):
        if overwrite:
            os.remove(output_path)
        else:
            raise FileExistsError(f"Output file '{output_path}' already exists.")
    start_time = time.time()

    with (
        core.ShowerDataFile(input_path, "r") as input_file,
        core.ShowerDataFile(output_path, "w", shape=input_file.shape) as output_file,
    ):
        num_showers = len(input_file)
        for start_idx in range(0, num_showers, batch_size):
            end_idx = min(start_idx + batch_size, num_showers)
            input_showers = input_file[start_idx:end_idx]
            filtered_showers = filter_showers(
                shower_data=input_showers,
                radius=radius,
                ecal_threshold=ecal_threshold,
                hcal_threshold=hcal_threshold,
                num_layers_ecal=num_layers_ecal,
            )
            output_file[start_idx:end_idx] = filtered_showers
            if verbose and (end_idx % (10 * batch_size) == 0 or end_idx == num_showers):
                print(
                    f"[{time.time() - start_time:6.1f}s]: Processed {end_idx} / {num_showers} showers."
                )


def get_detector_thresholds(
    path: str | os.PathLike[str],
) -> tuple[float, float, int]:
    """\
    Get the energy thresholds for ECAL and HCAL from the detector module. Determines which detector
    to use from the metadata in the specified shower data file.

    Args:
        path: Path to the shower data file.

    Returns:
        A tuple containing the ECAL and HCAL energy thresholds (in GeV).
    """
    metadata = core.load_metadata(path)
    detector_name = metadata.get("clustered_for", "ILD_unclustered")
    if detector_name == "ILD":
        geometry = detector.get_ILD_geometry()
    elif detector_name == "ILD_unclustered":
        geometry = detector.get_ILD_unclustered_geometry()
    else:
        return 0.0, 0.0, 0

    return (
        geometry.ecal_threshold / 1e3,
        geometry.hcal_threshold / 1e3,
        geometry.num_layers_ecal,
    )


def add_parser_arguments(parser: argparse.ArgumentParser) -> None:
    """Add filter-related arguments to an argparse parser.

    Args:
        parser: An argparse parser.
    """
    parser.add_argument(
        "-r",
        "--radius",
        type=float,
        default=float("inf"),
        help="Radius (in millimeters) for the cylindrical cut filter.",
    )
    parser.add_argument(
        "--ecal-threshold",
        type=float,
        default=0.0,
        help="Energy threshold (in MeV) for the ECAL hit filter.",
    )
    parser.add_argument(
        "--hcal-threshold",
        type=float,
        default=0.0,
        help="Energy threshold (in MeV) for the HCAL hit filter.",
    )
    parser.add_argument(
        "-d",
        "--use-detector-thresholds",
        action="store_true",
        help="Use the detector's predefined energy thresholds for filtering hits.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input file containing showers to be filtered.",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output file to save the filtered showers.",
    )


def parse_arguments(args: list[str] | None = None) -> argparse.Namespace:
    """\
    Parse command line arguments for the filtering module.

    Args:
        args: List of command line arguments. If None, uses sys.argv.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Filter hits in shower data based on spatial and energy criteria."
    )
    add_parser_arguments(parser)
    return parser.parse_args(args=args)


def main(args: argparse.Namespace) -> None:
    print("Filter Shower Data with the following parameters:")
    print("\n".join(f"{k}: {v}" for k, v in vars(args).items()))
    print()

    if args.use_detector_thresholds:
        ecal_threshold, hcal_threshold, num_layers_ecal = get_detector_thresholds(
            args.input
        )
        print("Using detector thresholds:")
        print(f"  ECAL threshold: {ecal_threshold * 1e3:.2f} MeV")
        print(f"  HCAL threshold: {hcal_threshold * 1e3:.2f} MeV")
        print(f"  Number of ECAL layers: {num_layers_ecal}")
        print()
    else:
        ecal_threshold = args.ecal_threshold / 1e3  # Convert MeV to GeV
        hcal_threshold = args.hcal_threshold / 1e3
        num_layers_ecal = 30
    try:
        filter_file(
            input_path=args.input,
            output_path=args.output,
            radius=args.radius,
            ecal_threshold=ecal_threshold,
            hcal_threshold=hcal_threshold,
            num_layers_ecal=num_layers_ecal,
            overwrite=args.overwrite,
            verbose=True,
        )
    except FileExistsError:
        warnings.warn(
            f"Output file '{args.output}' already exists. Use --overwrite to overwrite it.",
            UserWarning,
        )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
