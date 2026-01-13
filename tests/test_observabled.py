import os
import re
import tempfile

import h5py
import numpy as np
import pytest

import showerdata
from showerdata import detector


def init_num_points_per_layer_test_showers() -> showerdata.Showers:
    showers = showerdata.Showers(
        energies=np.zeros((5, 1)),
        pdg=np.full(5, 11, dtype=np.int32),
        points=np.zeros((5, 10, 5)),
    )
    showers.points[..., 2] = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [99, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 0],
        ],
        dtype=np.float32,
    )
    for i, num_points in enumerate([5, 1, 10, 0, 9]):
        showers.points[i, :num_points, 3] = 42.0
    return showers


def test_calc_num_points_per_layer_basic() -> None:
    showers = init_num_points_per_layer_test_showers()

    result = showerdata.observables.calc_num_points_per_layer(showers)
    assert isinstance(result, np.ndarray), "Result should be a numpy array"
    assert result.dtype == np.int32, "Result should be int32"
    assert len(result) == 5, "Expected 5 showers"
    assert result.shape[1] == 100, "Expected 100 layers (max layer is 99)"

    expected_first = np.zeros(100, dtype=np.int32)
    for layer in [0, 1, 2, 3, 4]:
        expected_first[layer] = 1
    np.testing.assert_array_equal(
        result[0], expected_first, "First shower layer counts incorrect"
    )

    expected_second = np.zeros(100, dtype=np.int32)
    expected_second[99] = 1
    np.testing.assert_array_equal(
        result[1], expected_second, "Second shower layer counts incorrect"
    )

    expected_third = np.zeros(100, dtype=np.int32)
    expected_third[5] = 10
    np.testing.assert_array_equal(
        result[2], expected_third, "Third shower layer counts incorrect"
    )

    np.testing.assert_array_equal(
        result[3], np.zeros(100, dtype=np.int32), "Fourth shower should have no points"
    )


def test_calc_num_points_per_layer_args() -> None:
    showers = init_num_points_per_layer_test_showers()

    result = showerdata.observables.calc_num_points_per_layer(showers)
    result_120 = showerdata.observables.calc_num_points_per_layer(
        showers, num_layers=120
    )
    assert result_120.shape == (5, 120), "Expected shape (5, 120)"
    np.testing.assert_array_equal(
        result_120[:, :100], result, "Results should match for overlapping layers"
    )
    np.testing.assert_array_equal(
        result_120[:, 100:],
        np.zeros((5, 20), dtype=np.int32),
        "Extra layers should be zero",
    )


def test_calc_num_points_per_layer_empty() -> None:
    empty_showers = showerdata.Showers(
        energies=np.zeros((2, 1)),
        pdg=np.array([11, 22], dtype=np.int32),
        points=np.zeros((2, 5, 5)),
    )
    result_empty = showerdata.observables.calc_num_points_per_layer(
        empty_showers, num_layers=10
    )
    assert result_empty.shape == (2, 10), "Expected shape (2, 10)"
    np.testing.assert_array_equal(
        result_empty,
        np.zeros((2, 10), dtype=np.int32),
        "Empty showers should give zero counts",
    )


def test_calc_num_points_per_layer_bounded() -> None:
    bounded_showers = showerdata.Showers(
        energies=np.array([[100.0], [200.0]]),
        pdg=np.array([11, 11], dtype=np.int32),
        points=np.zeros((2, 5, 5)),
    )
    bounded_showers.points[0, :3, 2] = [0, 1, 2]
    bounded_showers.points[0, :3, 3] = [10.0, 20.0, 30.0]
    bounded_showers.points[1, :3, 2] = [5, 6, 7]
    bounded_showers.points[1, :3, 3] = [5.0, 15.0, 25.0]

    result_bounded = showerdata.observables.calc_num_points_per_layer(
        bounded_showers, num_layers=10
    )
    assert result_bounded.shape == (2, 10), "Expected shape (2, 10)"
    assert result_bounded[0, 0] == 1, "First shower should have 1 point in layer 0"
    assert result_bounded[0, 1] == 1, "First shower should have 1 point in layer 1"
    assert result_bounded[0, 2] == 1, "First shower should have 1 point in layer 2"
    assert result_bounded[1, 5] == 1, "Second shower should have 1 point in layer 5"
    assert result_bounded[1, 6] == 1, "Second shower should have 1 point in layer 6"
    assert result_bounded[1, 7] == 1, "Second shower should have 1 point in layer 7"


def init_energy_per_layer_test_showers() -> showerdata.Showers:
    showers = showerdata.Showers(
        energies=np.array([[100.0], [200.0], [50.0], [0.0], [150.0]]),
        pdg=np.full(5, 11, dtype=np.int32),
        points=np.zeros((5, 10, 5)),
    )
    showers.points[..., 2] = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [99, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 0],
        ],
        dtype=np.float32,
    )
    energy_values: list[list[float]] = [
        [10.0, 15.0, 20.0, 25.0, 30.0],
        [42.0],
        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        [],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ]

    for i, energies in enumerate(energy_values):
        num_points = len(energies)
        showers.points[i, :num_points, 3] = energies

    return showers


def test_calc_energy_per_layer_basic() -> None:
    showers = init_energy_per_layer_test_showers()

    result = showerdata.observables.calc_energy_per_layer(showers)
    assert isinstance(result, np.ndarray), "Result should be a numpy array"
    assert result.dtype == np.float32, "Result should be float32"
    assert len(result) == 5, "Expected 5 showers"
    assert result.shape[1] == 100, "Expected 100 layers (max layer is 99)"

    expected_first = np.zeros(100, dtype=np.float32)
    for layer, energy in enumerate([10.0, 15.0, 20.0, 25.0, 30.0]):
        expected_first[layer] = energy
    np.testing.assert_allclose(
        result[0],
        expected_first,
        rtol=1e-6,
        err_msg="First shower energy sums incorrect",
    )

    expected_second = np.zeros(100, dtype=np.float32)
    expected_second[99] = 42.0
    np.testing.assert_allclose(
        result[1],
        expected_second,
        rtol=1e-6,
        err_msg="Second shower energy sums incorrect",
    )

    # Verify third shower (layer 5 should have total energy 50.0 = 10 * 5.0)
    expected_third = np.zeros(100, dtype=np.float32)
    expected_third[5] = 50.0
    np.testing.assert_allclose(
        result[2],
        expected_third,
        rtol=1e-6,
        err_msg="Third shower energy sums incorrect",
    )

    np.testing.assert_allclose(
        result[3],
        np.zeros(100, dtype=np.float32),
        rtol=1e-6,
        err_msg="Fourth shower should have no energy",
    )


def test_calc_energy_per_layer_args() -> None:
    showers = init_energy_per_layer_test_showers()

    result = showerdata.observables.calc_energy_per_layer(showers)
    result_120 = showerdata.observables.calc_energy_per_layer(showers, num_layers=120)
    assert result_120.shape == (5, 120), "Expected shape (5, 120)"

    np.testing.assert_allclose(
        result_120[:, :100],
        result,
        rtol=1e-6,
        err_msg="Results should match for overlapping layers",
    )
    np.testing.assert_allclose(
        result_120[:, 100:],
        np.zeros((5, 20), dtype=np.float32),
        rtol=1e-6,
        err_msg="Extra layers should be zero",
    )


def test_calc_energy_per_layer_multi() -> None:
    test_showers = showerdata.Showers(
        energies=np.array([[100.0]]),
        pdg=np.array([11], dtype=np.int32),
        points=np.zeros((1, 3, 5)),
    )
    test_showers.points[0, :3, 2] = [5, 5, 5]
    test_showers.points[0, :3, 3] = [10.0, 20.0, 30.0]

    result_sum = showerdata.observables.calc_energy_per_layer(
        test_showers, num_layers=10
    )
    expected_sum = np.zeros(10, dtype=np.float32)
    expected_sum[5] = 60.0  # 10 + 20 + 30
    np.testing.assert_allclose(
        result_sum[0],
        expected_sum,
        rtol=1e-6,
        err_msg="Energy summation within layer incorrect",
    )


def test_calc_energy_per_layer_empty() -> None:
    empty_showers = showerdata.Showers(
        energies=np.zeros((2, 1)),
        pdg=np.array([11, 22], dtype=np.int32),
        points=np.zeros((2, 5, 5)),
    )
    result_empty = showerdata.observables.calc_energy_per_layer(
        empty_showers, num_layers=10
    )
    assert result_empty.shape == (2, 10), "Expected shape (2, 10)"
    np.testing.assert_allclose(
        result_empty,
        np.zeros((2, 10), dtype=np.float32),
        rtol=1e-6,
        err_msg="Empty showers should give zero energy",
    )


def test_calc_energy_per_layer_bounded() -> None:
    bounded_showers = showerdata.Showers(
        energies=np.array([[100.0], [200.0]]),
        pdg=np.array([11, 11], dtype=np.int32),
        points=np.zeros((2, 5, 5)),
    )
    bounded_showers.points[0, :3, 2] = [0, 1, 2]
    bounded_showers.points[0, :3, 3] = [10.0, 20.0, 30.0]
    bounded_showers.points[1, :3, 2] = [5, 6, 7]
    bounded_showers.points[1, :3, 3] = [5.0, 15.0, 25.0]

    result_bounded = showerdata.observables.calc_energy_per_layer(
        bounded_showers, num_layers=10
    )
    assert result_bounded.shape == (2, 10), "Expected shape (2, 10)"
    np.testing.assert_allclose(
        result_bounded[0, 0],
        10.0,
        rtol=1e-6,
        err_msg="First shower should have 10.0 energy in layer 0",
    )
    np.testing.assert_allclose(
        result_bounded[0, 1],
        20.0,
        rtol=1e-6,
        err_msg="First shower should have 20.0 energy in layer 1",
    )
    np.testing.assert_allclose(
        result_bounded[0, 2],
        30.0,
        rtol=1e-6,
        err_msg="First shower should have 30.0 energy in layer 2",
    )
    np.testing.assert_allclose(
        result_bounded[1, 5],
        5.0,
        rtol=1e-6,
        err_msg="Second shower should have 5.0 energy in layer 5",
    )
    np.testing.assert_allclose(
        result_bounded[1, 6],
        15.0,
        rtol=1e-6,
        err_msg="Second shower should have 15.0 energy in layer 6",
    )
    np.testing.assert_allclose(
        result_bounded[1, 7],
        25.0,
        rtol=1e-6,
        err_msg="Second shower should have 25.0 energy in layer 7",
    )


def test_add_observables_to_file() -> None:
    shower_data = np.zeros((3, 10, 5), dtype=np.float32)

    shower_data[0, 0] = [0, 0, 0, 5.0, 0]
    shower_data[0, 1] = [1, 1, 1, 3.0, 0.1]
    shower_data[0, 2] = [2, 2, 2, 2.0, 0.2]

    shower_data[1, 0] = [0, 0, 5, 4.0, 0]
    shower_data[1, 1] = [1, 1, 10, 6.0, 0.1]

    shower_data[2, 0] = [0, 0, 7, 1.0, 0]
    shower_data[2, 1] = [1, 1, 7, 2.0, 0.1]
    shower_data[2, 2] = [2, 2, 7, 3.0, 0.2]
    shower_data[2, 3] = [3, 3, 7, 4.0, 0.3]

    showers = showerdata.Showers(
        energies=np.array([[10.0], [20.0], [30.0]]),
        pdg=np.array([11, 13, 22], dtype=np.int32),
        points=shower_data,
        num_points=np.array([3, 2, 4], dtype=np.int32),
    )

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
        temp_filename = temp_file.name

    try:
        showers.save(temp_filename, overwrite=True)

        showerdata.observables.add_observables_to_file(
            temp_filename, batch_size=2, detector_config=detector.get_test_geometry(20)
        )

        file = h5py.File(temp_filename, "r")
        try:
            num_points_data = np.array(file["observables/num_points_per_layer"])
            energy_data = np.array(file["observables/energy_per_layer"])

            assert num_points_data.shape == (3, 20)
            assert energy_data.shape == (3, 20)
            assert num_points_data.dtype == np.int32
            assert energy_data.dtype == np.float32

            assert num_points_data[0, 0] == 1, "Shower 0 should have 1 point in layer 0"
            assert num_points_data[0, 1] == 1, "Shower 0 should have 1 point in layer 1"
            assert num_points_data[0, 2] == 1, "Shower 0 should have 1 point in layer 2"
            assert np.isclose(energy_data[0, 0], 5.0), (
                "Shower 0 should have 5.0 energy in layer 0"
            )
            assert np.isclose(energy_data[0, 1], 3.0), (
                "Shower 0 should have 3.0 energy in layer 1"
            )
            assert np.isclose(energy_data[0, 2], 2.0), (
                "Shower 0 should have 2.0 energy in layer 2"
            )

            assert num_points_data[1, 5] == 1, "Shower 1 should have 1 point in layer 5"
            assert num_points_data[1, 10] == 1, (
                "Shower 1 should have 1 point in layer 10"
            )
            assert np.isclose(energy_data[1, 5], 4.0), (
                "Shower 1 should have 4.0 energy in layer 5"
            )
            assert np.isclose(energy_data[1, 10], 6.0), (
                "Shower 1 should have 6.0 energy in layer 10"
            )

            assert num_points_data[2, 7] == 4, (
                "Shower 2 should have 4 points in layer 7"
            )
            assert np.isclose(energy_data[2, 7], 10.0), (
                "Shower 2 should have 10.0 energy in layer 7"
            )

        finally:
            file.close()

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


def test_load_observables() -> None:
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
        temp_filename = temp_file.name

    test_data: dict[str, np.ndarray] = {
        "energies": np.array([[100.0], [50.0], [30.0]]),
        "pdg": np.array([11, 22, 11], dtype=np.int32),
        "directions": np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        "shower_ids": np.array([0, 1, 2], dtype=np.int32),
        "num_points_per_layer": np.array(
            [[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 0, 4]]
        ),
        "energy_per_layer": np.array(
            [
                [5.0, 3.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 4.0, 6.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 9.0],
            ]
        ),
    }

    try:
        with h5py.File(temp_filename, "w") as file:
            for key, data in test_data.items():
                if key in ["num_points_per_layer", "energy_per_layer"]:
                    group = file.require_group("observables")
                    group.create_dataset(key, data=data)
                else:
                    file.create_dataset(key, data=data)
        result = showerdata.observables.read_observables_from_file(temp_filename)
        assert isinstance(result, dict), "Result should be a dictionary"
        for key, expected in test_data.items():
            if key in ["energies", "pdg", "directions"]:
                key = "incident_" + key
            assert key in result, f"Missing key '{key}' in result"
            np.testing.assert_allclose(
                result[key],
                expected,
                rtol=1e-6,
                err_msg=f"Data mismatch for key '{key}'",
            )
        assert "total_energy" in result, "Missing computed 'total_energy'"
        assert "total_num_points" in result, "Missing computed 'total_num_points'"
        np.testing.assert_allclose(
            result["total_energy"],
            np.array([10.0, 10.0, 9.0]),
            rtol=1e-6,
            err_msg="Data mismatch for key 'total_energy'",
        )
        np.testing.assert_allclose(
            result["total_num_points"],
            np.array([3, 2, 4]),
            rtol=1e-6,
            err_msg="Data mismatch for key 'total_num_points'",
        )
        with h5py.File(temp_filename, "a") as file:
            del file["observables/energy_per_layer"]
        showerdata.observables.read_observables_from_file(temp_filename)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Observable 'energy_per_layer' not found in file '{temp_filename}'."
            ),
        ):
            showerdata.observables.read_observables_from_file(
                temp_filename,
                observables=["energy_per_layer"],
            )
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


def test_calc_energy_per_radial_bin_basic() -> None:
    showers = showerdata.Showers(
        points=[
            [
                [0.0, 0.0, 0.0, 10.0],
                [3.0, 4.0, 1.0, 5.0],
                [6.0, 8.0, 2.0, 3.0],
            ]
        ],
        energies=[18.0],
        pdg=22,
    )
    result = showerdata.observables.calc_energy_per_radial_bin(
        showers, bin_edges=[0, 5, 10, 15]
    )
    assert result.shape == (1, 3), f"Expected shape (1, 3), got {result.shape}"
    np.testing.assert_allclose(result[0], [10.0, 5.0, 3.0], rtol=1e-6)


def test_calc_energy_per_radial_bin_empty() -> None:
    empty = showerdata.Showers(
        points=np.zeros((2, 0, 4), dtype=np.float32), energies=[0.0, 0.0], pdg=11
    )
    result_empty = showerdata.observables.calc_energy_per_radial_bin(
        empty, bin_edges=[0, 10, 20]
    )
    assert result_empty.shape == (2, 2)
    np.testing.assert_allclose(result_empty, np.zeros((2, 2)), rtol=1e-6)


def test_calc_energy_per_radial_outside() -> None:
    showers_outside = showerdata.Showers(
        points=[[[1000.0, 0.0, 0.0, 5.0]]], energies=[5.0], pdg=22
    )
    result_outside = showerdata.observables.calc_energy_per_radial_bin(
        showers_outside, bin_edges=[0, 10, 20]
    )
    np.testing.assert_allclose(result_outside[0], [0.0, 0.0], rtol=1e-6)


def test_calc_center_of_energy_symmetric() -> None:
    showers = showerdata.Showers(
        points=[
            [
                [1.0, 0.0, 0.0, 10.0],
                [-1.0, 0.0, 0.0, 10.0],
                [0.0, 2.0, 0.0, 10.0],
                [0.0, -2.0, 0.0, 10.0],
            ]
        ],
        energies=[40.0],
        pdg=22,
    )
    result = showerdata.observables.calc_center_of_energy(showers)
    assert result.shape == (1, 3), f"Expected shape (1, 3), got {result.shape}"
    np.testing.assert_allclose(result[0], [0.0, 0.0, 0.0], rtol=1e-6)


def test_calc_center_of_energy_weighted() -> None:
    showers_weighted = showerdata.Showers(
        points=[
            [
                [10.0, 20.0, 30.0, 100.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
        energies=[101.0],
        pdg=11,
    )
    result_weighted = showerdata.observables.calc_center_of_energy(showers_weighted)
    expected = np.array(
        [100.0 * 10.0 / 101.0, 100.0 * 20.0 / 101.0, 100.0 * 30.0 / 101.0]
    )
    np.testing.assert_allclose(result_weighted[0], expected, rtol=1e-5)


def test_calc_center_of_energy_multiple() -> None:
    multi = showerdata.Showers(
        points=[
            [[1.0, 0.0, 0.0, 10.0], [3.0, 0.0, 0.0, 10.0]],
            [[0.0, 2.0, 5.0, 5.0], [0.0, 4.0, 5.0, 5.0]],
        ],
        energies=[20.0, 10.0],
        pdg=[22, 11],
    )
    result_multi = showerdata.observables.calc_center_of_energy(multi)
    assert result_multi.shape == (2, 3)
    np.testing.assert_allclose(result_multi[0], [2.0, 0.0, 0.0], rtol=1e-6)
    np.testing.assert_allclose(result_multi[1], [0.0, 3.0, 5.0], rtol=1e-6)
