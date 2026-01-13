import tempfile
from pathlib import Path

import numpy as np
import pytest

from showerdata import core, filter


def test_filter_showers_radius() -> None:
    points = np.array(
        [
            [[0, 0, 5, 1], [100, 0, 10, 2], [500, 0, 15, 3], [0, 0, 0, 0]],
            [[50, 50, 5, 1.5], [1000, 0, 10, 2.5], [0, 0, 0, 0], [0, 0, 0, 0]],
        ],
        dtype=np.float32,
    )
    showers = core.Showers(points, energies=[10.0, 20.0], pdg=22)

    filtered = filter.filter_showers(showers, radius=200.0)

    assert filtered.points[0, 0, 3] == 1.0
    assert filtered.points[0, 1, 3] == 2.0
    assert filtered.points[0, 2, 3] == 0.0
    assert filtered.points[1, 0, 3] == 1.5
    assert filtered.points[1, 1, 3] == 0.0


def test_filter_showers_energy_thresholds() -> None:
    points = np.array(
        [
            [[0, 0, 5, 0.5], [0, 0, 10, 2.0], [0, 0, 35, 0.8], [0, 0, 35, 3.0]],
        ],
        dtype=np.float32,
    )
    showers = core.Showers(points, energies=[100.0], pdg=22)

    filtered = filter.filter_showers(
        showers, ecal_threshold=1.0, hcal_threshold=1.5, num_layers_ecal=30
    )

    assert filtered.points[0, 0, 3] == 2.0
    assert filtered.points[0, 1, 3] == 3.0
    assert filtered.points[0, 2, 3] == 0.0
    assert filtered.points[0, 3, 3] == 0.0


def test_filter_keeps_padding_at_end() -> None:
    points = np.array(
        [
            [[0, 0, 5, 1], [0, 0, 10, 0.5], [0, 0, 15, 2], [0, 0, 0, 0]],
        ],
        dtype=np.float32,
    )
    showers = core.Showers(points, energies=[50.0], pdg=22)

    filtered = filter.filter_showers(showers, ecal_threshold=0.8)

    assert filtered.points[0, 0, 3] == 1.0
    assert filtered.points[0, 1, 3] == 2.0
    assert filtered.points[0, 2, 3] == 0.0
    assert filtered.points[0, 3, 3] == 0.0


def test_filter_file_basic() -> None:
    points = np.array(
        [
            [[0, 0, 5, 1], [100, 0, 10, 2], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[50, 50, 5, 1.5], [200, 0, 15, 0.5], [0, 0, 0, 0], [0, 0, 0, 0]],
        ],
        dtype=np.float32,
    )
    showers = core.Showers(points, energies=[[10.0], [20.0]], pdg=22)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.h5"
        output_path = Path(tmpdir) / "output.h5"

        showers.save(input_path)
        filter.filter_file(input_path, output_path, radius=150.0, ecal_threshold=0.6)

        result = core.load(output_path)
        assert result.points[0, 0, 3] == 1.0
        assert result.points[0, 1, 3] == 2.0
        assert result.points[1, 0, 3] == 1.5
        assert result.points[1, 1, 3] == 0.0


def test_filter_file_overwrite_error() -> None:
    points = np.array([[[0, 0, 5, 1], [0, 0, 0, 0]]], dtype=np.float32)
    showers = core.Showers(points, energies=[[10.0]], pdg=22)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.h5"
        output_path = Path(tmpdir) / "output.h5"

        showers.save(input_path)
        showers.save(output_path)

        with pytest.raises(FileExistsError):
            filter.filter_file(input_path, output_path)


def test_filter_file_overwrite_allowed() -> None:
    points = np.array([[[0, 0, 5, 1], [0, 0, 0, 0]]], dtype=np.float32)
    showers = core.Showers(points, energies=[[10.0]], pdg=22)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.h5"
        output_path = Path(tmpdir) / "output.h5"

        showers.save(input_path)
        showers.save(output_path)
        filter.filter_file(input_path, output_path, overwrite=True)

        result = core.load(output_path)
        assert len(result) == 1
        assert result.points[0, 0, 3] == 1.0


def test_get_detector_thresholds() -> None:
    points = np.array([[[0, 0, 5, 1], [0, 0, 0, 0]]], dtype=np.float32)
    showers = core.Showers(points, energies=[[10.0]], pdg=22)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.h5"
        showers.save(path)
        core.add_metadata(
            path,
            {"clustered_for": "ILD"},
        )

        ecal_thresh, hcal_thresh, num_layers_ecal = filter.get_detector_thresholds(path)
        assert isinstance(ecal_thresh, float)
        assert isinstance(hcal_thresh, float)
        assert isinstance(num_layers_ecal, int)
        assert ecal_thresh >= 0.0
        assert hcal_thresh >= 0.0
        assert num_layers_ecal >= 0


def test_parse_arguments() -> None:
    args = filter.parse_arguments(["input.h5", "output.h5"])
    assert args.input == "input.h5"
    assert args.output == "output.h5"
    assert args.radius == float("inf")
    assert args.ecal_threshold == 0.0

    args = filter.parse_arguments(
        ["--radius", "100", "--ecal-threshold", "0.5", "in.h5", "out.h5"]
    )
    assert args.radius == 100.0
    assert args.ecal_threshold == 0.5


def test_main() -> None:
    points = np.array([[[0, 0, 5, 1], [500, 0, 10, 2]]], dtype=np.float32)
    showers = core.Showers(points, energies=[[10.0]], pdg=22)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.h5"
        output_path = Path(tmpdir) / "output.h5"

        showers.save(input_path)
        args = filter.parse_arguments(
            ["--radius", "200", str(input_path), str(output_path)]
        )
        filter.main(args)

        result = core.load(output_path)
        assert result.points[0, 0, 3] == 1.0
        assert result.points[0, 1, 3] == 0.0
