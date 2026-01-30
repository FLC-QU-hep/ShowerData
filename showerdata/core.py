"""Core functionality for ShowerData library."""

import os
from collections.abc import Iterable, Iterator
from types import EllipsisType

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._version import __version__

index_type = int | slice | EllipsisType | ArrayLike


class IncidentParticles:
    """Data structure for incident particle data.

    Args:
        energies (ArrayLike): Energies of the incident particles.
        pdg (ArrayLike or int): Particle Data Group identifier(s).
        directions (Optional[ArrayLike]): Directions of the incident particles as a unit vector. Defaults to (0, 0, 1).

    Attributes:
        energies (NDArray): Energies of the incident particles.
        directions (NDArray): Directions of the incident particles given as a unit vector.
        pdg (NDArray): Particle Data Group identifiers for the incident particles.
    """

    def __init__(
        self,
        energies: ArrayLike,
        pdg: ArrayLike | int,
        directions: ArrayLike | None = None,
    ):
        self.energies: NDArray[np.float32]
        self.pdg: NDArray[np.int32]
        self.directions: NDArray[np.float32]

        self.energies = np.asarray(energies, dtype=np.float32)
        self.pdg = np.asarray(pdg, dtype=np.int32)
        if self.pdg.ndim == 0:
            self.pdg = self.pdg.reshape((1,)).repeat(len(self.energies))
        if directions is not None:
            self.directions = np.asarray(directions, dtype=np.float32)
        else:
            self.directions = np.zeros((len(self.energies), 3), dtype=np.float32)
            self.directions[:, 2] = 1.0
        if self.energies.ndim != 2 or self.energies.shape[1] != 1:
            raise ValueError(
                "Energies must be a 2D array with shape (num_particles, 1)."
            )
        if self.directions.ndim != 2 or self.directions.shape[1] != 3:
            raise ValueError(
                "Directions must be a 2D array with shape (num_particles, 3)."
            )
        if self.pdg.ndim != 1:
            raise ValueError("PDG must be a 1D array.")
        if (
            self.energies.shape[0] != self.directions.shape[0]
            or self.energies.shape[0] != self.pdg.shape[0]
        ):
            raise ValueError("All input arrays must have the same length.")

    def __repr__(self) -> str:
        return f"IncidentParticles(energies={self.energies}, pdg={self.pdg}, directions={self.directions})"

    def __len__(self) -> int:
        return self.energies.shape[0]

    def __getitem__(self, index: index_type) -> "IncidentParticles":
        """Get a subset of the IncidentParticles."""
        if isinstance(index, int):
            index = slice(index, index + 1)
        elif (isinstance(index, tuple) and index == ()) or isinstance(
            index, EllipsisType
        ):
            index = Ellipsis
        elif not isinstance(index, slice):
            index = np.asarray(index).astype(np.int64)
            if index.ndim != 1:
                raise ValueError("Multi-dimensional indexing is not supported.")

        return IncidentParticles(
            energies=self.energies[index],
            pdg=self.pdg[index],
            directions=self.directions[index],
        )

    def save(self, path: str | os.PathLike[str], overwrite: bool = False) -> None:
        """
        Save incident particles data to an HDF5 file.

        Args:
            path (str | os.PathLike[str]): Path to the HDF5 file.
            overwrite (bool): If True, overwrite existing file. Defaults to False.
        """
        if not overwrite and os.path.isfile(path):
            raise FileExistsError(
                f"File {path} already exists. Use overwrite=True to overwrite."
            )
        with h5py.File(path, "w") as file:
            file.create_dataset("energies", data=self.energies)
            file.create_dataset("directions", data=self.directions)
            file.create_dataset("pdg", data=self.pdg)

            file.attrs["showerdata_version"] = __version__


class Showers:
    """Data structure for shower data.

    Args:
        points (ArrayLike): Shower point cloud.
        energies (ArrayLike): Energies of the incident particles.
        pdg (ArrayLike or int): Particle Data Group identifier(s).
        directions (Optional[ArrayLike]): Directions of the incident particles as a unit vector. Defaults to (0, 0, 1).
        shower_ids (Optional[ArrayLike]): Unique identifiers for each shower. Defaults to sequential IDs.
        copy (bool | None): If True, data will be copied to ensure immutability. Defaults to None.

    Attributes:
        points (NDArray): Array of shower points. Format: (num_showers, max_points, 4 or 5).
        energies (NDArray): Energies of the incident particles.
        directions (NDArray): Directions of the incident particles given as a unit vector.
        pdg (NDArray): Particle Data Group identifiers for the incident particles.
        shower_ids (NDArray): Unique identifiers for each shower.
    """

    def __init__(
        self,
        points: ArrayLike = (),
        energies: ArrayLike = (),
        pdg: ArrayLike | int = (),
        directions: ArrayLike | None = None,
        shower_ids: ArrayLike | None = None,
        num_points: ArrayLike | None = None,
        copy: bool | None = None,
    ) -> None:
        if isinstance(points, tuple) and len(points) == 0:
            points = np.empty((0, 0, 5), dtype=np.float32)
        points = np.asarray(points, dtype=np.float32, copy=copy)
        energies = np.asarray(energies, dtype=np.float32, copy=copy)
        if energies.ndim == 1:
            energies = energies.reshape((-1, 1))
        if isinstance(pdg, int):
            pdg = np.full((len(points),), pdg, dtype=np.int32)
        else:
            pdg = np.asarray(pdg, dtype=np.int32, copy=copy)
        if pdg.ndim > 1 and any(d > 1 for d in pdg.shape[1:]):
            raise ValueError("Invalid shape for PDG array.")
        pdg = pdg.reshape((-1,))
        if directions is None:
            directions = np.zeros((len(points), 3), dtype=np.float32)
            directions[:, 2] = 1.0
        else:
            directions = np.asarray(directions, dtype=np.float32, copy=copy)
        if shower_ids is None:
            shower_ids = np.arange(len(points), dtype=np.int32)
        else:
            shower_ids = np.asarray(shower_ids, dtype=np.int32, copy=copy)
        if num_points is None:
            num_points_array: NDArray[np.int32] = np.count_nonzero(
                points[:, :, 3], axis=1
            ).astype(np.int32)
        else:
            num_points_array = np.asarray(num_points, dtype=np.int32, copy=copy)
        self.points: NDArray[np.float32] = points
        self.energies: NDArray[np.float32] = energies
        self.pdg: NDArray[np.int32] = pdg
        self.directions: NDArray[np.float32] = directions
        self.shower_ids: NDArray[np.int32] = shower_ids
        self._num_points: NDArray[np.int32] = num_points_array
        self.__post_init__()

    def __post_init__(self):
        if self.points.ndim != 3 or self.points.shape[2] not in (4, 5):
            raise ValueError(
                f"Points must be a 3D array with shape (num_showers, max_points, 4 or 5) got {self.points.shape}"
            )
        if self.energies.ndim != 2 or self.energies.shape != (self.points.shape[0], 1):
            raise ValueError("Energies must be a 2D array with shape (num_showers, 1).")
        if self.directions.ndim != 2 or self.directions.shape != (
            self.points.shape[0],
            3,
        ):
            raise ValueError(
                "Directions must be a 2D array with shape (num_showers, 3)."
            )
        if self.pdg.ndim != 1 or self.pdg.shape[0] != self.points.shape[0]:
            raise ValueError("PDG must be a 1D array with length equal to num_showers.")
        if (
            self.shower_ids.ndim != 1
            or self.shower_ids.shape[0] != self.points.shape[0]
        ):
            raise ValueError(
                "Shower IDs must be a 1D array with length equal to num_showers."
            )
        if np.any(self.points[:, :, 3] < 0):
            raise ValueError("Point energies must be non-negative.")
        if (
            self._num_points.ndim != 1
            or self._num_points.shape[0] != self.points.shape[0]
        ):
            raise ValueError(
                "num_points must be a 1D array with length equal to num_showers."
            )
        if np.any((self.points[:, 1:, 3] > 0) & ~(self.points[:, :1, 3] > 0)):
            raise ValueError("Padding should be in the end of the shower points.")

    def __len__(self) -> int:
        return self.points.shape[0]

    def __getitem__(self, index: index_type) -> "Showers":
        if isinstance(index, int):
            index = slice(index, index + 1)
        elif (isinstance(index, tuple) and index == ()) or isinstance(
            index, EllipsisType
        ):
            index = Ellipsis
        elif not isinstance(index, slice):
            index = np.asarray(index).astype(np.int64)
            if index.ndim != 1:
                raise ValueError("Multi-dimensional indexing is not supported.")

        return Showers(
            points=self.points[index],
            energies=self.energies[index],
            pdg=self.pdg[index],
            directions=self.directions[index],
            shower_ids=self.shower_ids[index],
            num_points=self._num_points[index],
        )

    def __iter__(self) -> Iterator["Showers"]:
        return iter(self[i] for i in range(len(self)))

    @property
    def is_empty(self) -> bool:
        """
        Check if the Showers instance is empty.
        """
        return self.points.shape[0] == 0

    def __add__(self, other: "Showers") -> "Showers":
        if not isinstance(other, Showers):
            raise TypeError("Can only add Showers instances.")
        if self.is_empty:
            return other.copy()
        if other.is_empty:
            return self.copy()
        if self.points.shape[2] != other.points.shape[2]:
            raise ValueError("Incompatible shower shapes.")
        if self.points.shape[1] < other.points.shape[1]:
            self_showers = np.pad(
                self.points,
                ((0, 0), (0, other.points.shape[1] - self.points.shape[1]), (0, 0)),
                mode="constant",
            )
        else:
            self_showers = self.points
        if self_showers.shape[1] > other.points.shape[1]:
            other_showers = np.pad(
                other.points,
                ((0, 0), (0, self_showers.shape[1] - other.points.shape[1]), (0, 0)),
                mode="constant",
            )
        else:
            other_showers = other.points
        return Showers(
            points=np.concatenate([self_showers, other_showers], axis=0),
            energies=np.concatenate([self.energies, other.energies], axis=0),
            pdg=np.concatenate([self.pdg, other.pdg], axis=0),
            directions=np.concatenate([self.directions, other.directions], axis=0),
            shower_ids=np.concatenate([self.shower_ids, other.shower_ids], axis=0),
            num_points=np.concatenate([self._num_points, other._num_points], axis=0),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Showers):
            return False
        return (
            np.array_equal(self.points, other.points)
            and np.array_equal(self.energies, other.energies)
            and np.array_equal(self.pdg, other.pdg)
            and np.array_equal(self.directions, other.directions)
            and np.array_equal(self.shower_ids, other.shower_ids)
            and np.array_equal(self._num_points, other._num_points)
        )

    def copy(self) -> "Showers":
        """
        Create a copy of the Showers instance.
        Returns:
            Showers: A new Showers instance with copied data.
        """
        return Showers(
            points=self.points.copy(),
            energies=self.energies.copy(),
            pdg=self.pdg.copy(),
            directions=self.directions.copy(),
            shower_ids=self.shower_ids.copy(),
            num_points=self._num_points.copy(),
        )

    def save(self, path: str | os.PathLike[str], overwrite: bool = False) -> None:
        """
        Save data to an HDF5 file.

        Args:
            path (str | os.PathLike[str]): Path to the HDF5 file.
            overwrite (bool): If True, overwrite existing file. Defaults to False.
        """
        if not overwrite and os.path.isfile(path):
            raise FileExistsError(
                f"File {path} already exists. Use overwrite=True to overwrite."
            )
        showers_list = [
            shower[:num_points].flatten()
            for shower, num_points in zip(self.points, self._num_points)
        ]
        vlen_dtype = h5py.vlen_dtype(np.dtype("float32"))

        with h5py.File(path, "w") as file:
            # circumvent h5py issue with vlen data
            # https://github.com/h5py/h5py/issues/2429
            showers_dataset = file.create_dataset(
                "showers",
                shape=(len(self.points),),
                dtype=vlen_dtype,
            )
            showers_dataset[...] = showers_list
            file.create_dataset("energies", data=self.energies)
            file.create_dataset("directions", data=self.directions)
            file.create_dataset("pdg", data=self.pdg)
            file.create_dataset("shower_ids", data=self.shower_ids)
            file.create_dataset("shape", data=self.points.shape)
            file.create_dataset("num_points", data=self._num_points)

            file.attrs["showerdata_version"] = __version__

    @property
    def inc_particles(self) -> IncidentParticles:
        """
        Get the incident particles associated with the showers.
        """
        return IncidentParticles(
            energies=self.energies,
            pdg=self.pdg,
            directions=self.directions,
        )


def concatenate(data: Iterable[Showers]) -> Showers:
    """
    Concatenate a iterable of Showers instances into a single instance.

    Args:
        data (Iterable[Showers]): Iterable of Showers instances to concatenate.

    Returns:
        Showers: Concatenated Showers instance.
    """
    data = [s for s in data if not s.is_empty]
    if len(data) == 0:
        return Showers()
    max_points = max(s.points.shape[1] for s in data)
    points = [
        np.pad(s.points, ((0, 0), (0, max_points - s.points.shape[1]), (0, 0)))
        for s in data
    ]
    return Showers(
        points=np.concatenate(points, axis=0),
        energies=np.concatenate([s.energies for s in data], axis=0),
        pdg=np.concatenate([s.pdg for s in data], axis=0),
        directions=np.concatenate([s.directions for s in data], axis=0),
        shower_ids=np.concatenate([s.shower_ids for s in data], axis=0),
        num_points=np.concatenate([s._num_points for s in data], axis=0),
    )


def get_file_shape(path: str | os.PathLike[str]) -> tuple[int, int, int]:
    """
    Get the shape of the showers dataset in an HDF5 file. Only works for files
    containing shower data.

    Args:
        path (str | os.PathLike[str]): Path to the HDF5 file.

    Returns:
        tuple[int, int, int]: Shape of the showers dataset (num_showers, num_points, 4).
    """
    with h5py.File(path) as h5file:
        if "shape" not in h5file:
            raise ValueError(f"{path} is not a valid shower data file. (no shape key)")
        shape_dataset = h5file["shape"]
        if not isinstance(shape_dataset, h5py.Dataset):
            raise ValueError(
                f"{path} is not a valid shower data file. (shape is not a dataset)"
            )
        if (
            shape_dataset.ndim != 1
            or shape_dataset.shape[0] != 3
            or shape_dataset.dtype.kind != "i"
        ):
            raise ValueError(
                f"{path} is not a valid shower data file. (invalid shape dataset)"
            )
        shape = tuple(shape_dataset[...].tolist())
    return shape


def get_file_length(path: str | os.PathLike[str]) -> int:
    """
    Get the number of samples in an HDF5 shower data file. Unlike get_file_shape,
    this function works also for files only containing incident particle data.

    Args:
        path (str | os.PathLike[str]): Path to the HDF5 file.
    Returns:
        int: Number of samples in the file.
    """
    with h5py.File(path) as h5file:
        if "energies" not in h5file:
            raise ValueError(
                f"{path} is not a valid shower data file. (no energies dataset)"
            )
        energies_dataset = h5file["energies"]
        if not isinstance(energies_dataset, h5py.Dataset):
            raise ValueError(
                f"{path} is not a valid shower data file. (energies is not a dataset)"
            )
        length = energies_dataset.shape[0]
    return length


def save(
    data: Showers | IncidentParticles,
    path: str | os.PathLike[str],
    overwrite: bool = False,
) -> None:
    """
    Save data to an HDF5 file.

    Args:
        data (Showers | IncidentParticles): Data to save.
        path (str | os.PathLike[str]): Path to the HDF5 file.
        overwrite (bool): If True, overwrite existing file. Defaults to False.
    """
    data.save(path, overwrite)


def load_metadata(path: str | os.PathLike[str]) -> dict[str, str | int | float | bool]:
    """
    Load metadata from an HDF5 file.

    Args:
        path (str | os.PathLike[str]): Path to the HDF5 file.
    Returns:
        dict[str, str|int|float|bool]: Metadata attributes from the HDF5 file.
    """
    metadata: dict[str, str | int | float | bool] = {}
    with h5py.File(path, "r") as h5file:
        for key, value in h5file.attrs.items():
            metadata[key] = value
    return metadata


def add_metadata(
    path: str | os.PathLike[str], metadata: dict[str, str | int | float | bool]
) -> None:
    """
    Add metadata to an HDF5 file.

    Args:
        path (str | os.PathLike[str]): Path to the HDF5 file.
        metadata (dict[str, str|int|float|bool]): Metadata to add as attributes.
    """
    with h5py.File(path, "a") as h5file:
        for key, value in metadata.items():
            h5file.attrs[key] = value


def save_target(
    data: NDArray[np.float32],
    path: str | os.PathLike[str],
    num_points: NDArray[np.int32] | None = None,
    name: str = "target",
    overwrite: bool = False,
) -> None:
    """
    Save latent space target data for a specific generative model to an HDF5 file.
    The target usually has the same shape as the shower points.

    Args:
        data (NDArray[np.float32]): Target data to save.
        num_points (NDArray[np.int32] | None): Number of points for each shower in the target data.
        path (str | os.PathLike[str]): Path to the HDF5 file.
        name (str): Name of the target dataset in the HDF5 file. Defaults to "target".
        overwrite (bool): If True, overwrite existing dataset in file. Defaults to False.
    """
    if num_points is None and not os.path.isfile(path):
        raise ValueError("num_points must be provided if the file does not exist yet.")
    if data.ndim != 3:
        raise ValueError(
            "Data must be a 3D array with shape (num_showers, max_points, features)."
        )
    if num_points is None:
        with h5py.File(path, "r") as h5file:
            num_points = _get_int_data(h5file, "num_points")
    num_points = num_points.flatten()
    if num_points.shape[0] != data.shape[0]:
        raise ValueError(
            "num_points must have the same length as the number of showers in data."
        )
    if np.any(num_points < 0):
        raise ValueError("num_points values must be non-negative.")
    if np.any(num_points > data.shape[1]):
        raise ValueError(
            f"num_points values cannot exceed the maximum number of points ({data.shape[1]})."
        )
    point_clouds = [
        shower[:num_points_l].flatten().astype(np.float32)
        for shower, num_points_l in zip(data, num_points)
    ]
    vlen_dtype = h5py.vlen_dtype(np.dtype("float32"))
    with h5py.File(path, "a") as h5file:
        if name in h5file:
            if overwrite:
                del h5file[name]
            else:
                raise FileExistsError(
                    f"Dataset '{name}' already exists in {path}. Use overwrite=True to overwrite."
                )
        h5file.create_group(name)

        # circumvent h5py issue with vlen data
        # https://github.com/h5py/h5py/issues/2429
        point_cloud_dataset = h5file.create_dataset(
            name=name + "/point_clouds",
            shape=(len(point_clouds),),
            dtype=vlen_dtype,
        )
        point_cloud_dataset[...] = point_clouds
        h5file.create_dataset(
            name + "/shape", data=np.array(data.shape, dtype=np.int32)
        )
        h5file.create_dataset(name + "/num_points", data=num_points)


def _get_np_array(data: h5py.File, key: str, index: index_type = ...) -> np.ndarray:
    """Helper function to get a NumPy array from HDF5 file."""
    if key not in data:
        raise KeyError(f"Key '{key}' not found in HDF5 file.")
    dataset = data[key]
    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(f"Key '{key}' is not a dataset.")
    return dataset[index]


def _get_float_data(
    data_file: h5py.File, key: str, index: index_type = ...
) -> NDArray[np.float32]:
    """Helper function to get float data from HDF5 file."""
    data = _get_np_array(data_file, key, index)
    return np.asarray(data, dtype=np.float32)


def _get_int_data(
    data_file: h5py.File, key: str, index: index_type = ...
) -> NDArray[np.int32]:
    """Helper function to get integer data from HDF5 file."""
    data = _get_np_array(data_file, key, index)
    return np.asarray(data, dtype=np.int32)


def _get_shower_data(
    data_file: h5py.File,
    key: str = "showers",
    index: index_type = ...,
    max_points: int = -1,
) -> NDArray[np.float32]:
    """Helper function to get shower data from HDF5 file."""
    if key == "showers":
        shape_key = "shape"
        point_clouds_key = key
    else:
        point_clouds_key = key + "/point_clouds"
        shape_key = key + "/shape"

    shower_list = _get_np_array(data_file, point_clouds_key, index)
    shape = tuple(_get_int_data(data_file, shape_key).tolist())
    if max_points < 0:
        max_points = shape[1]
    showers = np.zeros(
        (len(shower_list), max_points, shape[2]),
        dtype=np.float32,
    )
    for i, shower in enumerate(shower_list):
        shower = shower.reshape(-1, shape[2])
        if len(shower) > max_points:
            shower = shower[:max_points]
        showers[i, : len(shower)] = shower
    return showers


def load(
    path: str | os.PathLike[str],
    start: int = 0,
    stop: int | None = None,
    max_points: int | None = -1,
) -> Showers:
    """
    Load shower data from an HDF5 file.

    Args:
        path (str | os.PathLike[str]): Path to the HDF5 file.
        start (int): Start index for loading showers. Defaults to 0.
        stop (Optional[int]): Stop index for loading showers. If None, load until end of file. Defaults to None.
        max_points (int): Maximum number of points to load per shower. If -1, load all points. Defaults to -1.

    Returns:
        Showers: Loaded shower data.
    """
    if max_points is None:
        max_points = -1
    with h5py.File(path, "r") as file:
        showers = Showers(
            points=_get_shower_data(file, "showers", slice(start, stop), max_points),
            energies=_get_float_data(file, "energies", slice(start, stop)),
            directions=_get_float_data(file, "directions", slice(start, stop)),
            pdg=_get_int_data(file, "pdg", slice(start, stop)),
            shower_ids=_get_int_data(file, "shower_ids", slice(start, stop)),
            num_points=_get_int_data(file, "num_points", slice(start, stop)),
        )
    return showers


def load_target(
    path: str | os.PathLike[str],
    key: str = "target",
    start: int = 0,
    stop: int | None = None,
    max_points: int | None = -1,
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """
    Load latent space target data for a specific generative model from an HDF5 file. The target usually has the same shape as the shower points.

    Args:
        path (str | os.PathLike[str]): Path to the HDF5 file.
        key (str): Name of the target dataset in the HDF5 file. Defaults to "target".
        start (int): Start index for loading target data. Defaults to 0.
        stop (Optional[int]): Stop index for loading target data. If None, load until end of file. Defaults to None.
        max_points (Optional[int]): Maximum number of points to load per shower. If -1, load all points. Defaults to -1.
    Returns:
        tuple[NDArray[np.float32], NDArray[np.int32]]: Loaded target data and corresponding number of points.
    """
    if max_points is None:
        max_points = -1

    with h5py.File(path) as file:
        target_data = _get_shower_data(file, key, slice(start, stop), max_points)
        num_points = _get_int_data(file, key + "/num_points", slice(start, stop))
    return target_data, num_points


def load_inc_particles(
    path: str | os.PathLike[str], start: int = 0, stop: int | None = None
) -> IncidentParticles:
    """
    Load incident particle data from an HDF5 file.

    Args:
        path (str | os.PathLike[str]): Path to the HDF5 file.
        start (int): Start index for loading incident particles. Defaults to 0.
        stop (Optional[int]): Stop index for loading incident particles. If None, load until end of file. Defaults to None.

    Returns:
        IncidentParticles: Loaded incident particle data.
    """
    with h5py.File(path, "r") as file:
        inc_particles = IncidentParticles(
            energies=_get_float_data(file, "energies", slice(start, stop)),
            pdg=_get_int_data(file, "pdg", slice(start, stop)),
            directions=_get_float_data(file, "directions", slice(start, stop)),
        )
    return inc_particles


def create_empty_file(
    path: str | os.PathLike[str], shape: tuple[int, int, int], overwrite: bool = True
) -> None:
    """
    Create an empty HDF5 file with the specified dataset shape. To be used before calling save_batch.

    Args:
        path (str | os.PathLike[str]): Path to the HDF5 file.
        shape (tuple[int, int, int]): Shape of the showers dataset.
        overwrite (bool): If True, overwrite existing file. Defaults to True.
    """
    vlen_dtype = h5py.vlen_dtype(np.dtype("float32"))
    if not overwrite and os.path.isfile(path):
        raise FileExistsError(
            f"File {path} already exists. Use overwrite=True to overwrite."
        )

    with h5py.File(path, "w") as file:
        file.create_dataset("showers", shape=(shape[0],), dtype=vlen_dtype)
        file.create_dataset("energies", shape=(shape[0], 1), dtype=np.float32)
        file.create_dataset("directions", shape=(shape[0], 3), dtype=np.float32)
        file.create_dataset("pdg", shape=(shape[0],), dtype=np.int32)
        file.create_dataset("shower_ids", shape=(shape[0],), dtype=np.int32)
        file.create_dataset("shape", data=shape, dtype=np.int32)
        file.create_dataset("num_points", shape=(shape[0],), dtype=np.int32)
        file.attrs["showerdata_version"] = __version__


def add_target_dataset(
    path: str | os.PathLike[str],
    shape: tuple[int, int, int],
    key: str = "target",
    exists_ok: bool = False,
) -> None:
    """
    Add an empty target dataset to an existing HDF5 file.

    Args:
        path (str | os.PathLike[str]): Path to the HDF5 file.
        shape (tuple[int, int, int]): Shape of the target dataset.
        key (str): Name of the target dataset in the HDF5 file. Defaults to "target".
        exists_ok (bool): If True, do not raise an error if the target dataset already exists. Defaults to False.
    """
    with h5py.File(path, "a") as file:
        if key in file:
            if exists_ok:
                return
            else:
                raise FileExistsError(f"Dataset '{key}' already exists in {path}.")
        file.create_group(key)
        vlen_dtype = h5py.vlen_dtype(np.dtype("float32"))
        file.create_dataset(key + "/point_clouds", shape=(shape[0],), dtype=vlen_dtype)
        file.create_dataset(key + "/shape", data=shape, dtype=np.int32)
        file.create_dataset(key + "/num_points", shape=(shape[0],), dtype=np.int32)


def _save_batch_to_dataset(
    data: np.ndarray | list,
    file: h5py.File,
    key: str,
    indexes: index_type = 0,
) -> None:
    """
    Save a batch of data to an HDF5 dataset.

    Args:
        data (NDArray): Data to save.
        file (h5py.File): Open HDF5 file.
        key (str): Dataset key in the file.
        indexes (index_type): Indexes to save the data to.
    """
    if key not in file:
        raise KeyError(f"Key '{key}' not found in HDF5 file.")
    dataset = file[key]
    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(f"Key '{key}' is not a dataset.")
    if isinstance(indexes, int):
        indexes = slice(indexes, indexes + len(data))
    dataset[indexes] = data


def save_batch(data: Showers, path: str | os.PathLike[str], start: int = 0) -> None:
    """
    Save a batch of shower data to an HDF5 file. The file must already exist and have the correct shape.
    Use create_empty_file to create the file first.

    Example:
        >>> showerdata.create_empty_file("showers.h5", shape=(1000, 500, 5))
        >>> # Now you can use save_batch to fill the file with data.
        >>> showers = showerdata.Showers(...)  # Create or load some showers
        >>> showerdata.save_batch(showers, "showers.h5", start=0)

    Args:
        data (Showers): Shower data to save.
        path (str | os.PathLike[str]): Path to the HDF5 file.
        start (int): Start index in the file. Defaults to 0.
    """
    showers_list = [
        shower[:num_points].flatten()
        for shower, num_points in zip(data.points, data._num_points)
    ]
    with h5py.File(path, "a") as file:
        if "shape" not in file:
            raise KeyError(f"Key 'shape' not found in {path}.")
        if not isinstance(file["shape"], h5py.Dataset):
            raise TypeError(f"Key 'shape' in {path} is not a dataset.")
        shape = tuple(_get_int_data(file, "shape").tolist())
        if not shape[1:] == data.points.shape[1:]:
            raise ValueError(
                f"Shape mismatch: expected {shape[1:]}, got {data.points.shape[1:]}"
            )
        if shape[0] < start + len(data):
            raise IndexError(
                f"Cannot write to {path}: start index {start} + data length {len(data)} exceeds file shape {shape[0]}"
            )
        _save_batch_to_dataset(showers_list, file, "showers", start)
        _save_batch_to_dataset(data.energies, file, "energies", start)
        _save_batch_to_dataset(data.directions, file, "directions", start)
        _save_batch_to_dataset(data.pdg, file, "pdg", start)
        _save_batch_to_dataset(data.shower_ids, file, "shower_ids", start)
        _save_batch_to_dataset(data._num_points, file, "num_points", start)


def save_target_batch(
    data: NDArray[np.float32],
    path: str | os.PathLike[str],
    num_points: NDArray[np.int32] | None = None,
    start: int = 0,
    key: str = "target",
) -> None:
    """
    Save a batch of latent space target data for a specific generative model to an HDF5 file.
    The target usually has the same shape as the shower points.
    The file must already exist and have the correct shape. Use add_target_dataset to create the target dataset first.

    Example:
        >>> showerdata.create_empty_file("showers.h5", shape=(1000, 500, 5))
        >>> showerdata.add_target_dataset("showers.h5", shape=(1000, 500, 3), key="target")
        >>> # Now you can use save_target_batch
        >>> target_data = np.random.rand(100, 500, 3).astype(np.float32)  # Example target data
        >>> num_points = np.random.randint(1, 501, size=(100,), dtype=np.int32)  # Example num_points
        >>> showerdata.save_target_batch(target_data, num_points, "showers.h5", start=0, key="target")

    Args:
        data (NDArray[np.float32]): Target data to save.
        path (str | os.PathLike[str]): Path to the HDF5 file.
        num_points (NDArray[np.int32] | None): Number of points for each shower in the target data.
        start (int): Start index in the file. Defaults to 0.
        key (str): Name of the target dataset in the HDF5 file. Defaults to "target".
    """
    if data.ndim != 3:
        raise ValueError(
            "Data must be a 3D array with shape (num_showers, max_points, features)."
        )
    if num_points is None:
        with h5py.File(path, "r") as h5file:
            num_points = _get_int_data(
                h5file, "num_points", slice(start, start + data.shape[0])
            )
    num_points = num_points.flatten()
    if num_points.shape[0] != data.shape[0]:
        raise ValueError(
            "num_points must have the same length as the number of showers in data."
        )
    if np.any(num_points < 0):
        raise ValueError("num_points values must be non-negative.")
    if np.any(num_points > data.shape[1]):
        raise ValueError(
            f"num_points values cannot exceed the maximum number of points ({data.shape[1]})."
        )

    point_clouds = [
        shower[:num_points_l].flatten().astype(np.float32)
        for shower, num_points_l in zip(data, num_points)
    ]
    with h5py.File(path, "a") as file:
        if key not in file:
            raise KeyError(f"Key '{key}' not found in {path}.")
        _save_batch_to_dataset(point_clouds, file, key + "/point_clouds", start)
        _save_batch_to_dataset(num_points, file, key + "/num_points", start)


class ShowerDataFile:
    """
    Context manager for handling shower data in HDF5 files.

    Example:
        >>> # read showers from a file
        >>> with showerdata.ShowerDataFile("showers.h5") as file:
        ...    print(file.shape)
        ...    print(file.num_showers)
        ...    shower = file[0]  # Get the first shower
        >>> # create a new file and write showers
        >>> with showerdata.ShowerDataFile(
        ...     path="new_showers.h5",
        ...     mode="w",
        ...     shape=(1000, 500, 5),
        ... ) as file:
        ...     new_showers = showerdata.Showers(...)  # Create or load some showers
        ...     file[0:100] = new_showers  # Write first 100 showers
        ...     file[100:200] = new_showers  # Write next 100 showers

    Args:
        path (str | os.PathLike[str]): Path to the HDF5 file.
        mode (str): File mode, either 'r' (read), 'w' (write), or 'a' (append). Defaults to 'r'.
        shape (Optional[tuple[int, int, int]]): Shape of the showers dataset when creating a new file. Required if mode is 'w'.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        mode: str = "r",
        shape: tuple[int, int, int] | None = None,
    ) -> None:
        self.path = path
        if mode not in ("r", "w", "a"):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'r', 'w', or 'a'.")
        if mode == "w":
            if shape is None:
                raise ValueError(
                    "Shape must be specified when opening a file for writing."
                )
            create_empty_file(path, shape, overwrite=True)
            mode = "a"
        self.file = h5py.File(path, mode)
        self.attrs = self.file.attrs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the HDF5 file."""
        self.file.close()

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the showers dataset in the file."""
        return tuple(_get_int_data(self.file, "shape").tolist())

    @property
    def num_showers(self) -> int:
        """Number of showers in the dataset."""
        return self.shape[0]

    def __len__(self) -> int:
        return self.num_showers

    def __getitem__(self, index: index_type) -> Showers:
        if not self.file:
            raise ValueError("File is not open.")
        if isinstance(index, tuple):
            if len(index) > 1 or len(index) == 0:
                raise ValueError("Multi-dimensional indexing is not supported.")
            else:
                index = index[0]
        if isinstance(index, int):
            if index < 0:
                index += self.num_showers
            if index < 0 or index >= self.num_showers:
                raise IndexError("Index out of range.")
            index = slice(index, index + 1)
        return Showers(
            points=_get_shower_data(self.file, "showers", index),
            energies=_get_float_data(self.file, "energies", index),
            directions=_get_float_data(self.file, "directions", index),
            pdg=_get_int_data(self.file, "pdg", index),
            shower_ids=_get_int_data(self.file, "shower_ids", index),
            num_points=_get_int_data(self.file, "num_points", index),
        )

    def __setitem__(self, index: index_type, data: Showers) -> None:
        if not self.file:
            raise ValueError("File is not open.")
        if self.file.mode not in ("r+", "w", "a"):
            raise ValueError(f"File is not open for writing (mode: {self.file.mode}).")
        if isinstance(index, tuple):
            if len(index) > 1 or len(index) == 0:
                raise ValueError("Multi-dimensional indexing is not supported.")
            else:
                index = index[0]
        if isinstance(index, int):
            if index < 0:
                index += self.num_showers
            if index < 0 or index >= self.num_showers:
                raise IndexError("Index out of range.")
            index = slice(index, index + 1)
        showers = [
            shower[:num_points].flatten()
            for shower, num_points in zip(data.points, data._num_points)
        ]
        _save_batch_to_dataset(showers, self.file, "showers", index)
        _save_batch_to_dataset(data.energies, self.file, "energies", index)
        _save_batch_to_dataset(data.directions, self.file, "directions", index)
        _save_batch_to_dataset(data.pdg, self.file, "pdg", index)
        _save_batch_to_dataset(data.shower_ids, self.file, "shower_ids", index)
        _save_batch_to_dataset(data._num_points, self.file, "num_points", index)

    def __iter__(self) -> Iterator["Showers"]:
        return iter(self[i] for i in range(len(self)))
