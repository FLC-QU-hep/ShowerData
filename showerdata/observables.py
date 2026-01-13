"""\
Provides functions to calculate various observables from shower data
and to add them to shower data files. This can be useful for analyzing
the fidelity of generative shower models or for training two-stage models
(e.g. layer-wise properties first, then point clouds).
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import h5py
import numpy as np
import numpy.typing as npt

from . import core, detector


def compute_threshold_mask(
    showers: core.Showers,
    detector_config: detector.DetectorGeometry | None = None,
) -> npt.NDArray[np.bool_]:
    """Compute a mask for the shower points based on detector thresholds.

    Args:
        showers (Showers): The showers to compute the mask for.
        detector_config (DetectorGeometry): The detector configuration.

    Returns:
        NDArray: A boolean mask indicating which points pass the thresholds.
    """
    if detector_config is None:
        return showers.points[..., 3] > 0
    ecal_mask = (showers.points[..., 2] < detector_config.num_layers_ecal - 0.5) & (
        showers.points[..., 3] > detector_config.ecal_threshold / 1e3
    )
    hcal_mask = (showers.points[..., 2] >= detector_config.num_layers_ecal - 0.5) & (
        showers.points[..., 3] > detector_config.hcal_threshold / 1e3
    )
    return ecal_mask | hcal_mask


def calc_num_points_per_layer(
    showers: core.Showers,
    num_layers: int = -1,
    detector_config: detector.DetectorGeometry | None = None,
) -> npt.NDArray[np.int32]:
    """Calculate the number of points per layer for each shower.

    Args:
        showers (Showers): The showers to calculate the number of points per layer for.
        num_layers (int, optional): The number of layers to consider. Defaults to -1 (infers from data).

    Returns:
        NDArray: A 2D array of shape (num_showers, num_layers) containing the number of points per layer for each shower.
    """
    num_showers = len(showers)
    layer_idx = (showers.points[..., 2] + 0.1).astype(np.int32)
    if num_layers < 0:
        num_layers = np.max(layer_idx).item() + 1
    points_per_layer = np.zeros((num_showers, num_layers), dtype=np.int32)
    shower_indices = (
        np.arange(num_showers).reshape(-1, 1).repeat(showers.points.shape[1], axis=1)
    )
    mask = compute_threshold_mask(showers, detector_config).astype(np.int32)
    np.add.at(points_per_layer, (shower_indices, layer_idx), mask)
    return points_per_layer


def calc_energy_per_layer(
    showers: core.Showers,
    num_layers: int = -1,
    detector_config: detector.DetectorGeometry | None = None,
) -> npt.NDArray[np.float32]:
    """Calculate the total energy per layer for each shower.

    Args:
        showers (Showers): The showers to calculate the total energy per layer for.

    Returns:
        NDArray: A 2D array of shape (num_showers, num_layers) containing the total energy per layer for each shower.
    """
    num_showers = len(showers)
    layer_idx = (showers.points[..., 2] + 0.1).astype(np.int32)
    if num_layers < 0:
        num_layers = np.max(layer_idx).item() + 1
    energy_per_layer = np.zeros((num_showers, num_layers), dtype=np.float32)
    shower_indices = (
        np.arange(num_showers).reshape(-1, 1).repeat(showers.points.shape[1], axis=1)
    )
    energies = showers.points[..., 3] * compute_threshold_mask(
        showers, detector_config
    ).astype(np.float32)
    np.add.at(energy_per_layer, (shower_indices, layer_idx), energies)
    return energy_per_layer


def calc_energy_per_radial_bin(
    showers: core.Showers,
    bin_edges: npt.ArrayLike | None = None,
    detector_config: detector.DetectorGeometry | None = None,
) -> npt.NDArray[np.float32]:
    """Calculate the total energy per radial bin for each shower.

    Args:
        showers (Showers): The showers to calculate the total energy per radial bin for.
        bin_edges (ArrayLike, optional): The edges of the radial bins. If None, defaults to 200 bins from 0 to 400 mm.

    Returns:
        NDArray: A 2D array of shape (num_showers, num_bins) containing the total energy per radial bin for each shower.
    """
    if bin_edges is None:
        bin_edges = np.linspace(0, 400, 201, dtype=np.float32)
    else:
        bin_edges = np.asarray(bin_edges, dtype=np.float32)
    num_showers = len(showers)
    radial_distances = np.sqrt(
        showers.points[..., 0] ** 2 + showers.points[..., 1] ** 2
    )
    bin_indices = np.digitize(radial_distances, bins=bin_edges) - 1
    num_bins = len(bin_edges) - 1
    energy_per_radial_bin = np.zeros((num_showers, num_bins), dtype=np.float32)
    shower_indices = (
        np.arange(num_showers).reshape(-1, 1).repeat(showers.points.shape[1], axis=1)
    )
    energies = showers.points[..., 3] * compute_threshold_mask(
        showers, detector_config
    ).astype(np.float32)
    energies[bin_indices == num_bins] = 0.0
    bin_indices[bin_indices == num_bins] = 0
    np.add.at(energy_per_radial_bin, (shower_indices, bin_indices), energies)
    return energy_per_radial_bin


def calc_center_of_energy(
    showers: core.Showers,
    detector_config: detector.DetectorGeometry | None = None,
) -> npt.NDArray[np.float32]:
    """Calculate the center of energy for each shower.

    Args:
        showers (Showers): The showers to calculate the center of energy for.

    Returns:
        NDArray: A 2D array of shape (num_showers, 3) containing the center of energy (x, y, z) for each shower.
    """
    energies = showers.points[..., 3] * compute_threshold_mask(
        showers, detector_config
    ).astype(np.float32)
    total_energies = np.sum(energies, axis=1, keepdims=True)
    weighted_positions = showers.points[..., :3] * energies[..., np.newaxis]
    sum_weighted_positions = np.sum(weighted_positions, axis=1)
    center_of_energy = sum_weighted_positions / (total_energies + 1e-8)
    return center_of_energy.astype(np.float32)


@dataclass
class ObservableInfo:
    key: str
    shape: tuple[int, ...]
    dtype: npt.DTypeLike
    function: Callable[[core.Showers], np.ndarray]


def _get_observables_list(
    num_showers: int,
    detector_config: detector.DetectorGeometry | None,
) -> list[ObservableInfo]:
    """Help function to get the list of available observables."""

    num_layers = detector_config.num_layers if detector_config else 78

    return [
        ObservableInfo(
            key="num_points_per_layer",
            shape=(num_showers, num_layers),
            dtype=np.int32,
            function=partial(
                calc_num_points_per_layer,
                num_layers=num_layers,
                detector_config=detector_config,
            ),
        ),
        ObservableInfo(
            key="energy_per_layer",
            shape=(num_showers, num_layers),
            dtype=np.float32,
            function=partial(
                calc_energy_per_layer,
                num_layers=num_layers,
                detector_config=detector_config,
            ),
        ),
        ObservableInfo(
            key="energy_per_radial_bin",
            shape=(num_showers, 200),
            dtype=np.float32,
            function=partial(
                calc_energy_per_radial_bin, detector_config=detector_config
            ),
        ),
        ObservableInfo(
            key="center_of_energy",
            shape=(num_showers, 3),
            dtype=np.float32,
            function=partial(calc_center_of_energy, detector_config=detector_config),
        ),
    ]


def add_observables_to_file(
    path: str | os.PathLike[str],
    batch_size: int = 1000,
    overwrite: bool = False,
    detector_config: detector.DetectorGeometry | None = None,
) -> None:
    """Calculate and add observables to an existing HDF5 file containing shower data.

    Args:
        path (str): The path to the HDF5 file.
        batch_size (int): The number of showers to process in each batch. Defaults to 1000.
        overwrite (bool): Whether to overwrite existing observables. Defaults to False.
    """
    num_showers = core.get_file_shape(path)[0]
    with h5py.File(path, "a") as file:
        clustered_for = file.attrs.get("clustered_for", "")
        if detector_config:
            pass
        elif clustered_for == "ILD":
            detector_config = detector.get_ILD_geometry()
        elif clustered_for == "":
            detector_config = detector.get_ILD_unclustered_geometry()
        else:
            raise ValueError(
                f"Unknown 'clustered_for' attribute value: {clustered_for}"
            )
        file.attrs["observables_computed_for"] = detector_config.name
        observables_list = _get_observables_list(num_showers, detector_config)
        if "observables" in file:
            if not overwrite:
                raise ValueError(
                    f"File {path} already contains an 'observables' group. Use overwrite option to replace it."
                )
            del file["observables"]
        group = file.create_group("observables")
        for obs in observables_list:
            group.create_dataset(
                obs.key,
                shape=obs.shape,
                dtype=obs.dtype,
            )
        for start in range(0, num_showers, batch_size):
            end = min(start + batch_size, num_showers)
            showers = core.Showers(
                points=core._get_shower_data(file, "showers", slice(start, end)),
                energies=core._get_float_data(file, "energies", slice(start, end)),
                pdg=core._get_int_data(file, "pdg", slice(start, end)),
                directions=core._get_float_data(file, "directions", slice(start, end)),
                shower_ids=core._get_int_data(file, "shower_ids", slice(start, end)),
            )
            for obs in observables_list:
                core._save_batch_to_dataset(
                    data=obs.function(showers),
                    file=file,
                    key=f"observables/{obs.key}",
                    indexes=start,
                )


def save_observables_to_file(
    path: str | os.PathLike[str],
    observables: dict[str, np.ndarray],
    overwrite: bool = False,
) -> None:
    """Save observables to an HDF5 file.

    Args:
        path (str): The path to the HDF5 file.
        observables (dict): A dictionary containing the observables to save.
        overwrite (bool, optional): Whether to overwrite the file if it exists. Defaults to False.
    """
    with h5py.File(path, "a") as file:
        groupe = file.require_group("observables")
        for key, data in observables.items():
            if key in groupe:
                if overwrite:
                    del groupe[key]
                else:
                    raise ValueError(
                        f"Observable '{key}' already exists in file '{path}'."
                    )
            groupe.create_dataset(key, data=data)


def _add_sum_based_observables(
    key: str,
    results: dict[str, np.ndarray],
    file: h5py.File,
    start: int = 0,
    stop: int | None = None,
) -> None:
    """Add sum-based observables to the results dictionary.
    If the observable is not in the results dictionary, it is calculated from the
    file.

    Args:
        key (str): The key of the observable to sum.
        results (dict): The dictionary containing the observables.
        file (h5py.File): The HDF5 file to read from.
        start (int, optional): The starting index of the showers to read. Defaults to 0.
        stop (int, optional): The stopping index of the showers to read. Defaults to None.
    """
    key_layer_wise = key.replace("total_", "") + "_per_layer"
    if key_layer_wise in results:
        results[key] = np.sum(results[key_layer_wise], axis=1)
    elif f"observables/{key_layer_wise}" in file:
        results[key] = np.sum(
            core._get_np_array(
                data=file,
                key=f"observables/{key_layer_wise}",
                index=slice(start, stop),
            ),
            axis=1,
        )


def read_observables_from_file(
    path: str | os.PathLike[str],
    observables: list[str] | None = None,
    start: int = 0,
    stop: int | None = None,
) -> dict[str, np.ndarray]:
    """Read observables from an HDF5 file. The default observables are:

    - num_points_per_layer
    - energy_per_layer
    - energy_per_radial_bin
    - center_of_energy
    - total_energy
    - total_num_points
    - incident_energies
    - incident_pdg
    - incident_directions
    - shower_ids

    Args:
        file (str): The HDF5 file to read from.
        observables (list[str], optional): The list of observables to read.
        start (int, optional): The starting index of the showers to read. Defaults to 0.
        stop (int, optional): The stopping index of the showers to read. If None, reads until the end. Defaults to None.

    Returns:
        dict: A dictionary containing the observables.
    """
    keys: dict[str, str | None] = {
        "num_points_per_layer": "observables/num_points_per_layer",
        "energy_per_layer": "observables/energy_per_layer",
        "energy_per_radial_bin": "observables/energy_per_radial_bin",
        "center_of_energy": "observables/center_of_energy",
        "total_energy": None,
        "total_num_points": None,
        "incident_energies": "energies",
        "incident_pdg": "pdg",
        "incident_directions": "directions",
        "shower_ids": "shower_ids",
    }
    if observables is None:
        observables = list(keys.keys())
        explicit_observables = False
    else:
        explicit_observables = True

    results: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as file:
        for obs in observables:
            key = keys.get(obs)
            if key is not None and key in file:
                results[obs] = core._get_np_array(file, key, slice(start, stop))
        for obs in observables:
            if obs.startswith("total_"):
                _add_sum_based_observables(
                    key=obs,
                    results=results,
                    file=file,
                    start=start,
                    stop=stop,
                )
    if explicit_observables:
        for obs in observables:
            if obs not in results:
                raise ValueError(f"Observable '{obs}' not found in file '{path}'.")

    return results


def read_point_energies(
    path: str | os.PathLike[str],
    start: int = 0,
    stop: int | None = None,
) -> npt.NDArray[np.float32]:
    """Read point energies from an HDF5 file.

    Args:
        path (str): The HDF5 file to read from.
        start (int, optional): The starting index of the showers to read. Defaults to 0.
        stop (int, optional): The stopping index of the showers to read. If None, reads until the end. Defaults to None.

    Returns:
        NDArray: A 1D array containing the point energies of all showers in the specified range.
    """
    with h5py.File(path, "r") as file:
        shower_list = core._get_np_array(file, "showers", slice(start, stop))
        shape = tuple(core._get_int_data(file, "shape").tolist())
    shower_list = [shower.reshape(-1, shape[2]) for shower in shower_list]
    point_energies = np.concatenate([shower[:, 3] for shower in shower_list], axis=0)
    return point_energies.astype(np.float32)
