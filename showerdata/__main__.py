import argparse

from . import cluster_module, detector, filter, observables, shift_showers, shuffle
from ._version import __version__


def add_observables_parser_options(parser: argparse.ArgumentParser) -> None:
    """Add options for the 'add_observables' command to the parser."""
    parser.add_argument(
        "filename",
        type=str,
        help="Path to the input HDF5 file",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for processing (default: 4096)",
    )
    parser.add_argument(
        "-l",
        "--num-layers",
        type=int,
        default=-1,
        help="Number of layers to process (default: 78)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing observables if they exist",
    )


def main():
    parser = argparse.ArgumentParser(description="Shower data format utility script")
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show program's version number and exit",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    shuffle_parser = subparsers.add_parser(
        "shuffle", help="Shuffle shower data from multiple input files"
    )
    shuffle.add_parser_options(shuffle_parser)
    observables_parser = subparsers.add_parser(
        "add-observables", help="Calculate and add observables to the shower data file"
    )
    add_observables_parser_options(observables_parser)
    shift_parser = subparsers.add_parser(
        "shift", help="Shift showers to undo incident angle effects"
    )
    shift_showers.add_parser_args(shift_parser)
    cluster_parser = subparsers.add_parser(
        "cluster", help="Cluster hits into readout cells using a regular grid"
    )
    cluster_module.add_parser_arguments(cluster_parser)
    filter_parser = subparsers.add_parser(
        "filter", help="Filter hits in shower data based on spatial and energy criteria"
    )
    filter.add_parser_arguments(filter_parser)

    args = parser.parse_args()
    if args.command == "shuffle":
        shuffle.main(args)
    elif args.command == "add-observables":
        try:
            observables.add_observables_to_file(
                path=args.filename,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
                detector_config=(
                    None
                    if args.num_layers == -1
                    else detector.get_test_geometry(args.num_layers)
                ),
            )
        except ValueError as e:
            # Existing observables and overwrite not set
            # Full stack trace not useful to the user in this case
            print(e)
            return
        except KeyError:
            print(f"Error: File {args.filename} does not contain valid shower data.")
            return
    elif args.command == "shift":
        shift_showers.main(args)
    elif args.command == "cluster":
        cluster_module.main(args)
    elif args.command == "filter":
        filter.main(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
