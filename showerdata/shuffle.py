"""Command-line tool to concatenate and shuffle shower data from multiple files and save to a new file."""

import argparse
import time

import numpy as np

import showerdata


def add_parser_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "input_files", nargs="+", help="Paths to one or more input HDF5 files"
    )
    parser.add_argument("output_file", type=str, help="Path to the output HDF5 file")
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for shuffling"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output file if it exists"
    )
    parser.add_argument(
        "--test",
        type=int,
        default=None,
        help="Test mode: load only the first N showers from each file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of showers to process in each batch (default: 1000)",
    )
    parser.add_argument(
        "--overwrite-ids",
        action="store_true",
        help="Assign new IDs to showers to avoid conflicts",
    )


def get_file_info(
    input_files: list[str], max_showers_per_file: int | None = None
) -> list[tuple[str, int]]:
    """Get information about input files without loading all data."""
    file_info: list[tuple[str, int]] = []

    for input_file in input_files:
        shape = showerdata.get_file_shape(input_file)
        total_showers = shape[0]

        if max_showers_per_file is not None:
            total_showers = min(total_showers, max_showers_per_file)

        file_info.append((input_file, total_showers))

    return file_info


def create_shuffled_indices(
    file_info: list[tuple[str, int]], seed: int | None = None
) -> list[tuple[str, int]]:
    """Create shuffled indices for all files."""
    if seed is not None:
        np.random.seed(seed)

    all_indices: list[tuple[str, int]] = []
    for file_path, num_showers in file_info:
        for i in range(num_showers):
            all_indices.append((file_path, i))

    indices_array = np.array(all_indices, dtype=object)
    permutation = np.random.permutation(len(indices_array))
    shuffled_indices = indices_array[permutation]

    return shuffled_indices.tolist()


def load_batch_by_indices(batch_indices: list[tuple[str, int]]) -> showerdata.Showers:
    """Load a batch of showers by their file and index. Shuffle result."""
    file_groups: dict[str, list[int]] = {}
    for file_path, index in batch_indices:
        if file_path not in file_groups:
            file_groups[file_path] = []
        file_groups[file_path].append(index)

    batch_showers: list[showerdata.Showers] = []
    for file_path, indices in file_groups.items():
        indices.sort()

        with showerdata.ShowerDataFile(file_path) as sdf:
            shower_data = sdf[indices]
            batch_showers.append(shower_data)

    combined = showerdata.concatenate(batch_showers)

    permutation = np.random.permutation(len(combined))
    combined = combined[permutation]

    return combined


def process_in_batches(
    shuffled_indices: list[tuple[str, int]],
    output_file: str,
    batch_size: int,
    shape: tuple[int, int, int],
    overwrite: bool = True,
    overwrite_ids: bool = False,
) -> None:
    """Process shuffled data in batches to minimize memory usage."""
    total_showers = len(shuffled_indices)

    showerdata.create_empty_file(output_file, shape=shape, overwrite=overwrite)

    for batch_start in range(0, total_showers, batch_size):
        batch_end = min(batch_start + batch_size, total_showers)
        batch_indices = shuffled_indices[batch_start:batch_end]

        print(
            f"[{time.strftime('%H:%M:%S')}]"
            f"Processing batch {batch_start // batch_size + 1}/{(total_showers + batch_size - 1) // batch_size} "
            f"(showers {batch_start + 1}-{batch_end})"
        )

        batch_data = load_batch_by_indices(batch_indices)
        if overwrite_ids:
            new_ids = np.arange(batch_start, batch_end, dtype=np.int32)
            batch_data.shower_ids = new_ids

        showerdata.save_batch(batch_data, output_file, start=batch_start)
        del batch_data  # save some memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shuffle shower data from multiple input files"
    )
    add_parser_options(parser)
    return parser.parse_args()


def main(args: argparse.Namespace):
    print(f"Analyzing {len(args.input_files)} input files...")
    file_info = get_file_info(args.input_files, max_showers_per_file=args.test)
    total_showers = sum(count for _, count in file_info)
    result_shape = (total_showers,) + showerdata.get_file_shape(file_info[0][0])[1:]

    if total_showers == 0:
        print("No showers to process!")
        return

    print(f"Total number of showers: {total_showers}")

    print(f"Processing data in batches of {args.batch_size} showers...")
    process_in_batches(
        shuffled_indices=create_shuffled_indices(file_info, seed=args.seed),
        output_file=args.output_file,
        batch_size=args.batch_size,
        shape=result_shape,
        overwrite=args.overwrite,
        overwrite_ids=args.overwrite_ids,
    )

    print("Done!")


if __name__ == "__main__":
    main(parse_args())
