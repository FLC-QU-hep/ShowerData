"""\
Functionality to shift showers in order to undo effects from incident angles.
Can be used to preprocess shower data before training generative models.
"""

import argparse
import math

import numpy as np
import numpy.typing as npt

from . import core, detector


def shift_layers(
    shower: npt.ArrayLike,
    direction: npt.ArrayLike,
    layer_bottom_pos: npt.ArrayLike,
    calo_surface: float,
    inverse: bool,
) -> npt.NDArray[np.float32]:
    shifted_shower = np.asarray(shower, dtype=np.float32, copy=True)
    direction = np.asarray(direction, dtype=np.float32)
    layer_bottom_pos = np.asarray(layer_bottom_pos, dtype=np.float32)
    z_coordinate = (
        layer_bottom_pos[(0.1 + shifted_shower[:, 2]).astype(int)] - calo_surface
    )
    if inverse:
        shifted_shower[:, 0] += direction[0] / direction[2] * z_coordinate
        shifted_shower[:, 1] += direction[1] / direction[2] * z_coordinate
    else:
        shifted_shower[:, 0] -= direction[0] / direction[2] * z_coordinate
        shifted_shower[:, 1] -= direction[1] / direction[2] * z_coordinate
    box = math.tan(math.pi / 8) * calo_surface
    mask = (
        (shifted_shower[:, 0] > -box)
        & (shifted_shower[:, 0] < box)
        & (shifted_shower[:, 1] > -box)
        & (shifted_shower[:, 1] < box)
    )
    result = np.zeros_like(shifted_shower)
    result[: np.count_nonzero(mask)] = shifted_shower[mask]
    return result


def process_files(
    input_file: str,
    output_file: str,
    layer_bottom_pos: npt.ArrayLike,
    calo_surface: float,
    inverse: bool,
    batch_size: int = 10000,
    verbose: bool = False,
) -> None:
    with (
        core.ShowerDataFile(input_file, "r") as file_in,
        core.ShowerDataFile(output_file, "w", file_in.shape) as file_out,
    ):
        for start in range(0, len(file_in), batch_size):
            end = min(start + batch_size, len(file_in))
            showers = file_in[start:end]
            shifted_showers = core.Showers(
                points=[
                    shift_layers(
                        shower, direction, layer_bottom_pos, calo_surface, inverse
                    )
                    for shower, direction in zip(showers.points, showers.directions)
                ],
                energies=showers.energies,
                directions=showers.directions,
                pdg=showers.pdg,
                shower_ids=showers.shower_ids,
            )
            file_out[start:end] = shifted_showers
            if verbose:
                print(f"Processed showers {start} to {end} of {len(file_in)}")


def add_parser_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input_file", type=str, help="Path to the input HDF5 file")
    parser.add_argument("output_file", type=str, help="Path to the output HDF5 file")
    parser.add_argument(
        "-i", "--inverse", action="store_true", help="Undo the shower shifting"
    )


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shift showers to undo incident angle effects"
    )
    add_parser_args(parser)
    return parser.parse_args(args)


def main(args: argparse.Namespace) -> None:
    config = detector.get_ILD_geometry()
    process_files(
        input_file=args.input_file,
        output_file=args.output_file,
        layer_bottom_pos=config.layer_bottom_pos,
        calo_surface=config.calo_surface,
        inverse=args.inverse,
    )


if __name__ == "__main__":
    main(parse_args())
