import numpy as np

import showerdata
from showerdata import Showers, detector


def test_cluster_empty_no_padding() -> None:
    showers = Showers(
        points=np.empty((2, 0, 4), dtype=np.float32),
        energies=np.array([10.0, 20.0], dtype=np.float32),
        pdg=22,
    )
    clustered_showers = showerdata.cluster(showers, random_shift=False)
    assert clustered_showers.points.shape == (2, 0, 4), (
        f"Expected shape (2, 0, 4), got {clustered_showers.points.shape}"
    )
    assert np.all(clustered_showers.energies == showers.energies), (
        "Energies do not match"
    )
    assert np.all(clustered_showers.pdg == showers.pdg), (
        f"PDG codes do not match (got {clustered_showers.pdg} but expected {showers.pdg})"
    )


def test_cluster_empty_with_padding() -> None:
    showers = Showers(
        points=np.zeros((3, 10, 4), dtype=np.float32),
        energies=np.array([5.0, 15.0, 25.0], dtype=np.float32),
        pdg=11,
    )
    clustered_showers = showerdata.cluster(showers, random_shift=True)
    assert clustered_showers.points.shape == (3, 10, 4), (
        f"Expected shape (3, 10, 4), got {clustered_showers.points.shape}"
    )
    assert np.all(clustered_showers.energies == showers.energies), (
        "Energies do not match"
    )
    assert np.all(clustered_showers.pdg == showers.pdg), (
        f"PDG codes do not match (got {clustered_showers.pdg} but expected {showers.pdg})"
    )
    assert np.all(clustered_showers.points == 0), "Points should remain zero"


def test_cluster_simple_case() -> None:
    detector_config = detector.DetectorGeometry(
        calo_surface=0.0,
        num_layers=20,
        ecal_cell_size=10.0,
        hcal_cell_size=20.0,
        num_layers_ecal=10,
        num_layers_hcal=10,
        layer_thickness_ecal=0.1,
        layer_thickness_hcal=0.2,
        layer_bottom_pos=tuple(range(20)),
    )
    showers = Showers(
        points=[
            [
                [5.0, 5.0, 0.0, 1.0],
                [15.0, 15.0, 5.0, 2.0],
                [25.0, 25.0, 10.0, 3.0],
                [35.0, 35.0, 15.0, 4.0],
                [45.0, 45.0, 19.0, 5.0],
            ],
            [
                [7.0, 7.0, 1.0, 1.5],
                [17.0, 17.0, 6.0, 2.5],
                [27.0, 27.0, 11.0, 3.5],
                [37.0, 37.0, 16.0, 4.5],
                [47.0, 47.0, 17.0, 5.5],
            ],
        ],
        energies=[15.0, 17.5],
        pdg=[22, -11],
    )
    expected_points = np.array(
        [
            [
                [5.0, 5.0, 0.0, 1.0],
                [15.0, 15.0, 5.0, 2.0],
                [30.0, 30.0, 10.0, 3.0],
                [30.0, 30.0, 15.0, 4.0],
                [50.0, 50.0, 19.0, 5.0],
            ],
            [
                [5.0, 5.0, 1.0, 1.5],
                [15.0, 15.0, 6.0, 2.5],
                [30.0, 30.0, 11.0, 3.5],
                [30.0, 30.0, 16.0, 4.5],
                [50.0, 50.0, 17.0, 5.5],
            ],
        ],
        dtype=np.float32,
    )
    clustered_showers = showerdata.cluster(
        showers, random_shift=False, detector_config=detector_config
    )
    assert clustered_showers.points.shape == (2, 5, 4), (
        f"Expected shape (2, 5, 4), got {clustered_showers.points.shape}"
    )
    assert np.allclose(clustered_showers.points, expected_points), (
        f"Points do not match expected values. Got: {clustered_showers.points} Expected: {expected_points}"
    )
    assert np.all(clustered_showers.energies == showers.energies), (
        "Energies do not match"
    )
    assert np.all(clustered_showers.pdg == showers.pdg), (
        f"PDG codes do not match (got {clustered_showers.pdg} but expected {showers.pdg})"
    )


def test_cluster_multiple_hits_per_cell() -> None:
    detector_config = detector.DetectorGeometry(
        calo_surface=0.0,
        num_layers=10,
        ecal_cell_size=10.0,
        hcal_cell_size=20.0,
        num_layers_ecal=5,
        num_layers_hcal=5,
        layer_thickness_ecal=0.1,
        layer_thickness_hcal=0.2,
        layer_bottom_pos=tuple(range(10)),
    )
    showers = Showers(
        points=[
            [
                [5.0, 5.0, 0.0, 1.0],
                [6.0, 6.0, 0.0, 2.0],
                [15.0, 15.0, 3.0, 3.0],
                [16.0, 16.0, 3.0, 4.0],
                [25.0, 0.0, 6.0, 5.0],
                [0.0, 26.0, 6.0, 6.0],
            ]
        ],
        energies=[21.0],
        pdg=[22],
    )
    expected_points = np.array(
        [
            [
                [5.0, 5.0, 0.0, 3.0],
                [15.0, 15.0, 3.0, 7.0],
                [10.0, 30.0, 6.0, 6.0],
                [30.0, 10.0, 6.0, 5.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=np.float32,
    )
    clustered_showers = showerdata.cluster(
        showers, random_shift=False, detector_config=detector_config
    )
    assert clustered_showers.points.shape == (1, 6, 4), (
        f"Expected shape (1, 6, 4), got {clustered_showers.points.shape}"
    )
    assert np.allclose(clustered_showers.points, expected_points), (
        "Points do not match expected values."
    )
    assert np.all(clustered_showers.energies == showers.energies), (
        "Energies do not match"
    )
    assert np.all(clustered_showers.pdg == showers.pdg), (
        f"PDG codes do not match (got {clustered_showers.pdg} but expected {showers.pdg})"
    )


def test_cluster_5d_points_with_time() -> None:
    detector_config = detector.DetectorGeometry(
        calo_surface=0.0,
        num_layers=10,
        ecal_cell_size=10.0,
        hcal_cell_size=20.0,
        num_layers_ecal=5,
        num_layers_hcal=5,
        layer_thickness_ecal=0.1,
        layer_thickness_hcal=0.2,
        layer_bottom_pos=tuple(range(10)),
    )
    showers = Showers(
        points=[
            [
                [5.0, 5.0, 0.0, 1.0, 1.5],
                [6.0, 6.0, 0.0, 2.0, 1.2],  # Same cell, earlier time
                [15.0, 15.0, 3.0, 3.0, 2.3],
                [16.0, 16.0, 3.0, 4.0, 2.8],  # Same cell, later time
                [25.0, 0.0, 6.0, 5.0, 3.5],
                [0.0, 26.0, 6.0, 6.0, 3.9],
            ]
        ],
        energies=[21.0],
        pdg=[22],
    )
    expected_points = np.array(
        [
            [
                [5.0, 5.0, 0.0, 3.0, 1.2],  # Energy summed, time is minimum
                [15.0, 15.0, 3.0, 7.0, 2.3],  # Energy summed, time is minimum
                [10.0, 30.0, 6.0, 6.0, 3.9],
                [30.0, 10.0, 6.0, 5.0, 3.5],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=np.float32,
    )
    clustered_showers = showerdata.cluster(
        showers, random_shift=False, detector_config=detector_config
    )
    assert clustered_showers.points.shape == (1, 6, 5), (
        f"Expected shape (1, 6, 5), got {clustered_showers.points.shape}"
    )
    assert np.allclose(clustered_showers.points, expected_points), (
        f"Points do not match expected values. Got: {clustered_showers.points} Expected: {expected_points}"
    )
    assert np.all(clustered_showers.energies == showers.energies), (
        "Energies do not match"
    )
    assert np.all(clustered_showers.pdg == showers.pdg), (
        f"PDG codes do not match (got {clustered_showers.pdg} but expected {showers.pdg})"
    )


def test_cluster_5d_no_hits() -> None:
    # Test empty showers with 5D format
    showers = Showers(
        points=np.empty((2, 0, 5), dtype=np.float32),
        energies=np.array([10.0, 20.0], dtype=np.float32),
        pdg=22,
    )
    clustered_showers = showerdata.cluster(showers, random_shift=False)
    assert clustered_showers.points.shape == (2, 0, 5), (
        f"Expected shape (2, 0, 5), got {clustered_showers.points.shape}"
    )
    assert np.all(clustered_showers.energies == showers.energies), (
        "Energies do not match"
    )
    assert np.all(clustered_showers.pdg == showers.pdg), (
        f"PDG codes do not match (got {clustered_showers.pdg} but expected {showers.pdg})"
    )
