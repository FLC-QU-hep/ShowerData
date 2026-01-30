import os
import re
import tempfile
from collections.abc import Generator

import h5py
import numpy as np
import pytest

import showerdata
from showerdata import Showers


def generate_test_data(
    batch_size: int = 100, num_batches: int = 10, max_points: int = 100
) -> Generator[Showers]:
    """Generate some dummy data for testing."""
    random_generator = np.random.default_rng(42)
    sample_pdgs = np.array([11, -11, 22, 211, -211])
    for _ in range(num_batches):
        result = np.zeros((batch_size, max_points, 5))
        num_points_all = random_generator.integers(1, max_points, size=batch_size)
        for i, num_points in enumerate(num_points_all):
            result[i, :num_points, 0] = random_generator.normal(0, 1, size=num_points)
            result[i, :num_points, 1] = random_generator.normal(0, 1, size=num_points)
            result[i, :num_points, 2] = random_generator.integers(
                0, 78, size=num_points
            ).astype(np.float64)
            result[i, :num_points, 3] = random_generator.uniform(
                1e-5, 2e0, size=num_points
            )
            result[i, :num_points, 4] = random_generator.uniform(
                0, 200, size=num_points
            )
        shower_data = showerdata.Showers(
            points=result.astype(np.float32),
            energies=random_generator.uniform(10.0, 100.0, size=(batch_size, 1)).astype(
                np.float32
            ),
            directions=random_generator.normal(0, 1, size=(batch_size, 3)).astype(
                np.float32
            ),
            pdg=random_generator.choice(sample_pdgs, size=batch_size).astype(np.int32),
            shower_ids=random_generator.integers(0, 1000, size=batch_size).astype(
                np.int32
            ),
            num_points=num_points_all.astype(np.int32),
        )
        yield shower_data


def test_save() -> None:
    os.makedirs("data", exist_ok=True)
    for data in generate_test_data():
        result = showerdata.save(data, "data/test.h5", overwrite=True)
        assert result is None, "Save function should return None"
        assert os.path.isfile("data/test.h5"), "File should be created"
        result = showerdata.save(data.inc_particles, "data/test_inc.h5", overwrite=True)
        assert result is None, "Save function should return None"
        assert os.path.isfile("data/test_inc.h5"), "File should be created"
        os.remove("data/test.h5")
        os.remove("data/test_inc.h5")


def test_create_showers() -> None:
    for data in generate_test_data():
        result = showerdata.Showers(
            points=data.points,
            energies=data.energies,
            pdg=11,
        )
        assert isinstance(result, Showers), "Result should be an instance of Showers"
        assert result.points.shape == data.points.shape, "Points shape mismatch"
        assert result.energies.shape == data.energies.shape, "Energies shape mismatch"
        assert result.pdg.shape == data.pdg.shape, "PDG shape mismatch"
        assert result.directions.shape == data.directions.shape, (
            "Directions shape mismatch"
        )
        assert result.shower_ids.shape == data.shower_ids.shape, (
            "Shower IDs shape mismatch"
        )


def test_concatenate() -> None:
    data = showerdata.concatenate(generate_test_data())
    assert isinstance(data, Showers), "Result should be an instance of Showers"
    assert data.points.shape[0] == 1000, "Points count mismatch"
    assert data.energies.shape[0] == 1000, "Energies count mismatch"
    assert data.pdg.shape[0] == 1000, "PDG count mismatch"
    assert data.directions.shape[0] == 1000, "Directions count mismatch"
    assert data.shower_ids.shape[0] == 1000, "Shower IDs count mismatch"


def test_incident_particle_creation() -> None:
    incident_particles = showerdata.IncidentParticles(
        energies=np.random.uniform(10.0, 100.0, size=(10, 1)).tolist(),
        pdg=11,
    )
    assert isinstance(incident_particles, showerdata.IncidentParticles), (
        "Result should be an instance of IncidentParticles"
    )
    assert isinstance(incident_particles.energies, np.ndarray), (
        "Energies should be a numpy array"
    )
    assert isinstance(incident_particles.pdg, np.ndarray), "PDG should be a numpy array"
    assert isinstance(incident_particles.directions, np.ndarray), (
        "Direction should be a numpy array"
    )
    assert incident_particles.energies.shape == (10, 1), "Energies shape mismatch"
    assert incident_particles.pdg.shape == (10,), "PDG shape mismatch"
    assert incident_particles.directions.shape == (10, 3), "Direction shape mismatch"


def test_incident_particle_invalid_directions() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("Directions must be a 2D array with shape (num_particles, 3)."),
    ):
        showerdata.IncidentParticles(
            energies=np.random.uniform(0, 1, size=(10, 1)),
            pdg=np.random.randint(0, 10, size=(10,)),
            directions=np.random.uniform(0, 1, size=(10, 4)),
        )


def test_incident_particle_invalid_energies() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("Energies must be a 2D array with shape (num_particles, 1)."),
    ):
        showerdata.IncidentParticles(
            energies=np.random.uniform(0, 1, size=(10, 2)),
            pdg=np.random.randint(0, 10, size=(10,)),
            directions=np.random.uniform(0, 1, size=(10, 3)),
        )


def test_incident_particle_invalid_pdg() -> None:
    with pytest.raises(ValueError, match=re.escape("PDG must be a 1D array.")):
        showerdata.IncidentParticles(
            energies=np.random.uniform(0, 1, size=(10, 1)),
            pdg=np.random.randint(0, 10, size=(10, 2)),
            directions=np.random.uniform(0, 1, size=(10, 3)),
        )


def test_incident_particle_mismatched_lengths() -> None:
    with pytest.raises(
        ValueError, match=re.escape("All input arrays must have the same length.")
    ):
        showerdata.IncidentParticles(
            energies=np.random.uniform(0, 1, size=(1, 1)),
            pdg=np.random.randint(0, 10, size=(10,)),
            directions=np.random.uniform(0, 1, size=(10, 3)),
        )


def test_save_and_load() -> None:
    os.makedirs("data", exist_ok=True)
    for data in generate_test_data():
        showerdata.save(data, "data/test_save_load.h5", overwrite=True)
        loaded_data = showerdata.load("data/test_save_load.h5")
        assert isinstance(loaded_data, Showers), (
            "Loaded data should be an instance of Showers"
        )
        assert np.array_equal(data.points, loaded_data.points), "Points data mismatch"
        assert np.array_equal(data.energies, loaded_data.energies), (
            "Energies data mismatch"
        )
        assert np.array_equal(data.pdg, loaded_data.pdg), "PDG data mismatch"
        assert np.array_equal(data.directions, loaded_data.directions), (
            "Directions data mismatch"
        )
        assert np.array_equal(data.shower_ids, loaded_data.shower_ids), (
            "Shower IDs data mismatch"
        )
        assert np.array_equal(data._num_points, loaded_data._num_points), (
            "Num points data mismatch"
        )
        os.remove("data/test_save_load.h5")


def test_save_and_load_inc() -> None:
    os.makedirs("data", exist_ok=True)
    for data in generate_test_data():
        showerdata.save(
            data.inc_particles, "data/test_save_load_inc.h5", overwrite=True
        )
        loaded_data = showerdata.load_inc_particles("data/test_save_load_inc.h5")
        assert isinstance(loaded_data, showerdata.IncidentParticles), (
            "Loaded data should be an instance of IncidentParticles"
        )
        assert np.array_equal(data.energies, loaded_data.energies), (
            "Energies data mismatch"
        )
        assert np.array_equal(data.pdg, loaded_data.pdg), "PDG data mismatch"
        assert np.array_equal(data.directions, loaded_data.directions), (
            "Directions data mismatch"
        )
    os.remove("data/test_save_load_inc.h5")


def test_load_single_shower() -> None:
    os.makedirs("data", exist_ok=True)
    data = next(generate_test_data(num_batches=1))
    showerdata.save(data, "data/test_load_single.h5", overwrite=True)
    loaded_data = showerdata.load("data/test_load_single.h5", start=0, stop=1)
    assert isinstance(loaded_data, Showers), (
        "Loaded data should be an instance of Showers"
    )
    assert np.array_equal(data.points[0], loaded_data.points[0]), "Points data mismatch"
    assert np.array_equal(data.energies[0], loaded_data.energies[0]), (
        "Energies data mismatch"
    )
    assert np.array_equal(data.pdg[0], loaded_data.pdg[0]), "PDG data mismatch"
    assert np.array_equal(data.directions[0], loaded_data.directions[0]), (
        "Directions data mismatch"
    )
    assert np.array_equal(data.shower_ids[0], loaded_data.shower_ids[0]), (
        "Shower IDs data mismatch"
    )
    assert np.array_equal(data._num_points[0], loaded_data._num_points[0]), (
        "Num points data mismatch"
    )
    os.remove("data/test_load_single.h5")


def test_save_in_batches() -> None:
    os.makedirs("data", exist_ok=True)
    batch_size = 5
    num_batches = 3
    max_points = 20
    showerdata.create_empty_file(
        "data/test_batch.h5", shape=(batch_size * num_batches, max_points, 5)
    )
    for i, data in enumerate(
        generate_test_data(
            batch_size=batch_size, num_batches=num_batches, max_points=max_points
        )
    ):
        result = showerdata.save_batch(data, "data/test_batch.h5", start=i * batch_size)
        assert result is None, "Save batch function should return None"
        assert os.path.isfile("data/test_batch.h5"), "Batch file should be created"
        loaded_data = showerdata.load(
            "data/test_batch.h5", start=i * batch_size, stop=(i + 1) * batch_size
        )
        assert isinstance(loaded_data, Showers), (
            "Loaded batch data should be an instance of Showers"
        )
        assert np.array_equal(data.points, loaded_data.points), (
            "Batch points data mismatch"
        )
        assert np.array_equal(data.energies, loaded_data.energies), (
            "Batch energies data mismatch"
        )
        assert np.array_equal(data.pdg, loaded_data.pdg), "Batch PDG data mismatch"
        assert np.array_equal(data.directions, loaded_data.directions), (
            "Batch directions data mismatch"
        )
        assert np.array_equal(data.shower_ids, loaded_data.shower_ids), (
            "Batch shower IDs data mismatch"
        )
        assert np.array_equal(data._num_points, loaded_data._num_points), (
            "Batch num points data mismatch"
        )
    os.remove("data/test_batch.h5")


def test_exceptions() -> None:
    os.makedirs("data", exist_ok=True)
    data = next(generate_test_data(batch_size=10, num_batches=1))
    shape = (5,) + data.points.shape[1:]
    showerdata.create_empty_file("data/test_out_of_bounds.h5", shape=shape)

    with pytest.raises(IndexError):
        showerdata.save_batch(data, "data/test_out_of_bounds.h5", start=0)

    with pytest.raises(FileExistsError):
        showerdata.create_empty_file(
            "data/test_out_of_bounds.h5", shape=shape, overwrite=False
        )

    shape = (20,) + data.points.shape[1:]
    showerdata.create_empty_file(
        "data/test_out_of_bounds.h5", shape=shape, overwrite=True
    )

    with pytest.raises(IndexError):
        showerdata.save_batch(data, "data/test_out_of_bounds.h5", start=15)
    os.remove("data/test_out_of_bounds.h5")


def test_copy_equal() -> None:
    shower_generator = generate_test_data(batch_size=10, num_batches=2)
    data = next(shower_generator)
    data_copy = data.copy()
    assert data_copy is not data, "Copy did not create a new instance"
    assert np.array_equal(data.points, data_copy.points), "Points data mismatch"
    assert data.points is not data_copy.points, "Copy did not create a new points array"
    assert data == data_copy, "Copy did not create an equal instance"
    new_data = next(shower_generator)
    assert data != new_data, "New data should not be equal to the original"


def test_slicing() -> None:
    data = next(generate_test_data(batch_size=10, num_batches=1))
    first_half = data[:5]
    assert len(first_half) == 5, "Slicing did not return the correct number of showers"
    assert np.array_equal(data.points[:5], first_half.points), (
        "Sliced points data mismatch"
    )
    second_half = data[5:]
    assert len(second_half) == 5, "Slicing did not return the correct number of showers"
    assert np.array_equal(data.points[5:], second_half.points), (
        "Sliced points data mismatch"
    )
    all = first_half + second_half
    assert len(all) == 10, (
        "Combining halves did not return the correct number of showers"
    )
    assert np.array_equal(data.points, all.points), "Combined points data mismatch"
    indexes = [1, 4, 6]
    sliced = data[indexes]
    assert len(sliced) == len(indexes), (
        "Slicing did not return the correct number of showers"
    )
    assert np.array_equal(data.points[indexes], sliced.points), (
        "Sliced points data mismatch"
    )
    indexes_array = np.array(indexes)
    sliced_array = data[indexes_array]
    assert len(sliced_array) == len(indexes), (
        "Slicing did not return the correct number of showers"
    )
    assert np.array_equal(data.points[indexes_array], sliced_array.points), (
        "Sliced points data mismatch"
    )
    data_all_slice = data[:]
    data_all_ellipsis = data[...]
    data_all_tuple = data[()]
    assert data == data_all_slice
    assert data == data_all_ellipsis
    assert data == data_all_tuple


def test_get_file_shape() -> None:
    os.makedirs("data", exist_ok=True)
    data = next(generate_test_data(batch_size=10, num_batches=1))
    showerdata.save(data, "data/test_shape.h5", overwrite=True)

    shape = showerdata.get_file_shape("data/test_shape.h5")
    assert shape == data.points.shape, "Unexpected shape for points dataset"

    os.remove("data/test_shape.h5")


def test_shower_file() -> None:
    os.makedirs("data", exist_ok=True)
    data = next(generate_test_data(batch_size=10, num_batches=1))
    showerdata.save(data, "data/test_shower_file.h5", overwrite=True)

    with showerdata.ShowerDataFile("data/test_shower_file.h5") as file:
        loaded_data = file[:]
    assert data == loaded_data, "Loaded data does not match original data"
    assert data.points.shape == loaded_data.points.shape, (
        "Loaded points data shape mismatch"
    )
    assert data.energies.shape == loaded_data.energies.shape, (
        "Loaded energies data shape mismatch"
    )
    assert data.pdg.shape == loaded_data.pdg.shape, "Loaded PDG data shape mismatch"
    assert data.directions.shape == loaded_data.directions.shape, (
        "Loaded directions data shape mismatch"
    )
    assert data.shower_ids.shape == loaded_data.shower_ids.shape, (
        "Loaded shower IDs data shape mismatch"
    )
    assert data._num_points.shape == loaded_data._num_points.shape, (
        "Loaded num_points data shape mismatch"
    )


def test_iterate_shower_file() -> None:
    data = next(generate_test_data(batch_size=10, num_batches=1))
    with tempfile.TemporaryDirectory() as temp_dir:
        file = os.path.join(temp_dir, "test_iterate_shower_file.h5")
        showerdata.save(data, file, overwrite=True)
        loaded_showers: list[showerdata.Showers] = []
        with showerdata.ShowerDataFile(file) as shower_file:
            for shower in shower_file:
                assert len(shower) == 1, "Each shower should contain a single entry"
                loaded_showers.append(shower)
        loaded_data = showerdata.concatenate(loaded_showers)
    assert np.array_equal(data.points, loaded_data.points), (
        "Loaded points data mismatch"
    )
    assert np.array_equal(data.energies, loaded_data.energies), (
        "Loaded energies data mismatch"
    )
    assert np.array_equal(data.pdg, loaded_data.pdg), "Loaded PDG data mismatch"
    assert np.array_equal(data.directions, loaded_data.directions), (
        "Loaded directions data mismatch"
    )
    assert np.array_equal(data.shower_ids, loaded_data.shower_ids), (
        "Loaded shower IDs data mismatch"
    )
    assert np.array_equal(data._num_points, loaded_data._num_points), (
        "Loaded num_points data mismatch"
    )


def generate_target_data(
    batch_size: int = 100, max_points: int = 100, features: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Generate target data for testing."""
    random_generator = np.random.default_rng(42)
    target_data = random_generator.uniform(
        0, 1, size=(batch_size, max_points, features)
    ).astype(np.float32)
    num_points = random_generator.integers(1, max_points + 1, size=batch_size).astype(
        np.int32
    )
    for i in range(batch_size):
        target_data[i, num_points[i] :, :] = 0.0  # Zero out unused points
    return target_data, num_points


def test_save_target() -> None:
    os.makedirs("data", exist_ok=True)
    target_data, num_points = generate_target_data(
        batch_size=10, max_points=50, features=3
    )

    showerdata.save_target(
        target_data, "data/test_target.h5", num_points, overwrite=True
    )
    assert os.path.isfile("data/test_target.h5"), "Target file should be created"

    showerdata.save_target(
        target_data,
        "data/test_target_custom.h5",
        num_points,
        name="custom_target",
        overwrite=True,
    )
    assert os.path.isfile("data/test_target_custom.h5"), (
        "Target file with custom name should be created"
    )

    os.remove("data/test_target.h5")
    os.remove("data/test_target_custom.h5")


def test_save_target_exceptions() -> None:
    os.makedirs("data", exist_ok=True)
    target_data, num_points = generate_target_data(
        batch_size=10, max_points=50, features=3
    )

    showerdata.save_target(
        target_data, "data/test_target_exists.h5", num_points, overwrite=True
    )
    with pytest.raises(FileExistsError):
        showerdata.save_target(
            target_data, "data/test_target_exists.h5", num_points, overwrite=False
        )

    bad_data = np.random.uniform(0, 1, size=(10, 50)).astype(np.float32)
    with pytest.raises(ValueError, match="Data must be a 3D array"):
        showerdata.save_target(
            bad_data, "data/test_target_bad.h5", num_points, overwrite=True
        )

    bad_num_points = np.random.randint(1, 51, size=5).astype(np.int32)
    with pytest.raises(ValueError, match="num_points must have the same length"):
        showerdata.save_target(
            target_data, "data/test_target_bad.h5", bad_num_points, overwrite=True
        )

    bad_data_4d = np.random.uniform(0, 1, size=(10, 50, 3, 2)).astype(np.float32)
    with pytest.raises(ValueError, match="Data must be a 3D array"):
        showerdata.save_target(
            bad_data_4d, "data/test_target_bad.h5", num_points, overwrite=True
        )

    bad_num_points_negative = np.array(
        [-1, 5, 10, 15, 20, 25, 30, 35, 40, 45], dtype=np.int32
    )
    with pytest.raises(ValueError, match="num_points values must be non-negative"):
        showerdata.save_target(
            target_data,
            "data/test_target_bad.h5",
            bad_num_points_negative,
            overwrite=True,
        )

    bad_num_points_too_large = np.array(
        [60, 55, 52, 51, 50, 49, 48, 47, 46, 45], dtype=np.int32
    )  # First value exceeds max_points=50
    with pytest.raises(
        ValueError, match="num_points values cannot exceed the maximum number of points"
    ):
        showerdata.save_target(
            target_data,
            "data/test_target_bad.h5",
            bad_num_points_too_large,
            overwrite=True,
        )

    os.remove("data/test_target_exists.h5")


def test_load_target() -> None:
    os.makedirs("data", exist_ok=True)
    target_data, num_points = generate_target_data(
        batch_size=20, max_points=50, features=3
    )

    showerdata.save_target(
        target_data, "data/test_load_target.h5", num_points, overwrite=True
    )

    loaded_data, loaded_num_points = showerdata.load_target("data/test_load_target.h5")
    assert isinstance(loaded_data, np.ndarray), "Loaded data should be a numpy array"
    assert isinstance(loaded_num_points, np.ndarray), (
        "Loaded num_points should be a numpy array"
    )
    assert loaded_data.shape == target_data.shape, (
        "Loaded data shape should match original"
    )
    assert np.array_equal(loaded_num_points, num_points), (
        "Loaded num_points should match original"
    )
    assert np.array_equal(loaded_data, target_data), "Loaded data should match original"

    showerdata.save_target(
        target_data,
        "data/test_load_target_custom.h5",
        num_points,
        name="custom_target",
        overwrite=True,
    )
    loaded_data_custom, loaded_num_points_custom = showerdata.load_target(
        "data/test_load_target_custom.h5", key="custom_target"
    )
    assert loaded_data_custom.shape == target_data.shape, (
        "Loaded custom data shape should match original"
    )
    assert np.array_equal(loaded_num_points_custom, num_points), (
        "Loaded custom num_points should match original"
    )
    assert np.array_equal(loaded_data_custom, target_data), (
        "Loaded custom data should match original"
    )

    loaded_data_slice, loaded_num_points_slice = showerdata.load_target(
        "data/test_load_target.h5", start=5, stop=15
    )
    expected_data_slice = target_data[5:15]
    expected_num_points_slice = num_points[5:15]
    assert loaded_data_slice.shape == expected_data_slice.shape, (
        "Sliced data shape should match expected"
    )
    assert np.array_equal(loaded_num_points_slice, expected_num_points_slice), (
        "Sliced num_points should match expected"
    )
    assert np.array_equal(loaded_data_slice, expected_data_slice), (
        "Sliced data should match expected"
    )

    loaded_data_limited, _ = showerdata.load_target(
        "data/test_load_target.h5", max_points=30
    )
    assert loaded_data_limited.shape[1] == 30, "Data should be limited to max_points"
    assert loaded_data_limited.shape[0] == target_data.shape[0], (
        "Number of showers should remain the same"
    )
    assert loaded_data_limited.shape[2] == target_data.shape[2], (
        "Number of features should remain the same"
    )
    assert np.array_equal(loaded_data_limited, target_data[:, :30, :]), (
        "Data should be correctly cut to max_points"
    )

    os.remove("data/test_load_target.h5")
    os.remove("data/test_load_target_custom.h5")


def test_load_target_exceptions() -> None:
    os.makedirs("data", exist_ok=True)

    with pytest.raises(FileNotFoundError):
        showerdata.load_target("data/non_existent.h5")

    data = next(generate_test_data(batch_size=5, num_batches=1))
    showerdata.save(data, "data/test_no_target.h5", overwrite=True)
    with pytest.raises(KeyError):
        showerdata.load_target("data/test_no_target.h5")

    target_data, num_points = generate_target_data(
        batch_size=5, max_points=20, features=3
    )
    showerdata.save_target(
        target_data, "data/test_wrong_key.h5", num_points, overwrite=True
    )
    with pytest.raises(KeyError):
        showerdata.load_target("data/test_wrong_key.h5", key="wrong_key")

    os.remove("data/test_no_target.h5")
    os.remove("data/test_wrong_key.h5")


def test_add_target_dataset() -> None:
    os.makedirs("data", exist_ok=True)

    data = next(generate_test_data(batch_size=10, num_batches=1))
    showerdata.save(data, "data/test_add_target.h5", overwrite=True)

    target_shape = (10, 50, 3)
    showerdata.add_target_dataset("data/test_add_target.h5", target_shape)

    with h5py.File("data/test_add_target.h5", "r") as file:
        assert "target" in file, "Target group should be created"
        assert "target/point_clouds" in file, (
            "Target point_clouds dataset should be created"
        )
        assert "target/shape" in file, "Target shape dataset should be created"
        assert "target/num_points" in file, (
            "Target num_points dataset should be created"
        )

        shape_dataset = file["target/shape"]
        shape_data = np.array(shape_dataset)
        assert tuple(shape_data.tolist()) == target_shape, (
            "Target shape should match specified shape"
        )

    showerdata.add_target_dataset(
        "data/test_add_target.h5", target_shape, key="custom_target"
    )
    with h5py.File("data/test_add_target.h5", "r") as file:
        assert "custom_target" in file, "Custom target group should be created"

    showerdata.add_target_dataset(
        "data/test_add_target.h5", target_shape, exists_ok=True
    )

    os.remove("data/test_add_target.h5")


def test_add_target_dataset_exceptions() -> None:
    os.makedirs("data", exist_ok=True)

    data = next(generate_test_data(batch_size=5, num_batches=1))
    showerdata.save(data, "data/test_add_target_exc.h5", overwrite=True)

    target_shape = (5, 20, 3)
    showerdata.add_target_dataset("data/test_add_target_exc.h5", target_shape)

    with pytest.raises(FileExistsError, match="Dataset 'target' already exists"):
        showerdata.add_target_dataset(
            "data/test_add_target_exc.h5", target_shape, exists_ok=False
        )

    os.remove("data/test_add_target_exc.h5")


def test_save_target_batch() -> None:
    os.makedirs("data", exist_ok=True)

    total_size = 20
    batch_size = 5
    max_points = 30
    features = 3
    target_shape = (total_size, max_points, features)

    showerdata.create_empty_file(
        "data/test_target_batch.h5", shape=(total_size, max_points, 5)
    )
    showerdata.add_target_dataset("data/test_target_batch.h5", target_shape)

    all_target_data: list[np.ndarray] = []
    all_num_points: list[np.ndarray] = []
    for i in range(0, total_size, batch_size):
        target_data, num_points = generate_target_data(
            batch_size=batch_size, max_points=max_points, features=features
        )
        all_target_data.append(target_data)
        all_num_points.append(num_points)

        showerdata.save_target_batch(
            target_data, "data/test_target_batch.h5", num_points, start=i
        )

    loaded_data, loaded_num_points = showerdata.load_target("data/test_target_batch.h5")

    expected_data = np.concatenate(all_target_data, axis=0)
    expected_num_points = np.concatenate(all_num_points, axis=0)

    assert loaded_data.shape == expected_data.shape, (
        "Loaded batch data shape should match expected"
    )
    assert np.array_equal(loaded_num_points, expected_num_points), (
        "Loaded batch num_points should match expected"
    )

    showerdata.add_target_dataset(
        "data/test_target_batch.h5", target_shape, key="batch_target"
    )
    target_data, num_points = generate_target_data(
        batch_size=batch_size, max_points=max_points, features=features
    )
    showerdata.save_target_batch(
        target_data,
        "data/test_target_batch.h5",
        num_points,
        start=0,
        key="batch_target",
    )

    loaded_data_custom, loaded_num_points_custom = showerdata.load_target(
        "data/test_target_batch.h5", key="batch_target", stop=batch_size
    )
    assert loaded_data_custom.shape == target_data.shape, (
        "Custom key batch data should match"
    )
    assert np.array_equal(loaded_num_points_custom, num_points), (
        "Custom key batch num_points should match"
    )

    os.remove("data/test_target_batch.h5")


def test_save_target_batch_exceptions() -> None:
    os.makedirs("data", exist_ok=True)

    showerdata.create_empty_file("data/test_target_batch_exc.h5", shape=(10, 20, 5))

    target_data, num_points = generate_target_data(
        batch_size=5, max_points=20, features=3
    )

    with pytest.raises(KeyError, match="Key 'target' not found"):
        showerdata.save_target_batch(
            target_data, "data/test_target_batch_exc.h5", num_points
        )

    showerdata.add_target_dataset("data/test_target_batch_exc.h5", (10, 20, 3))
    with pytest.raises(KeyError, match="Key 'wrong_key' not found"):
        showerdata.save_target_batch(
            target_data, "data/test_target_batch_exc.h5", num_points, key="wrong_key"
        )

    bad_data_2d = np.random.uniform(0, 1, size=(5, 20)).astype(np.float32)
    with pytest.raises(ValueError, match="Data must be a 3D array"):
        showerdata.save_target_batch(
            bad_data_2d, "data/test_target_batch_exc.h5", num_points
        )

    bad_num_points = np.random.randint(1, 21, size=3).astype(np.int32)
    with pytest.raises(ValueError, match="num_points must have the same length"):
        showerdata.save_target_batch(
            target_data, "data/test_target_batch_exc.h5", bad_num_points
        )

    bad_num_points_neg = np.array([-1, 5, 10, 15, 20], dtype=np.int32)
    with pytest.raises(ValueError, match="num_points values must be non-negative"):
        showerdata.save_target_batch(
            target_data, "data/test_target_batch_exc.h5", bad_num_points_neg
        )

    bad_num_points_large = np.array([25, 22, 21, 20, 19], dtype=np.int32)
    with pytest.raises(
        ValueError, match="num_points values cannot exceed the maximum number of points"
    ):
        showerdata.save_target_batch(
            target_data, "data/test_target_batch_exc.h5", bad_num_points_large
        )

    os.remove("data/test_target_batch_exc.h5")


def test_target_data_integration() -> None:
    os.makedirs("data", exist_ok=True)

    shower_data = next(generate_test_data(batch_size=10, num_batches=1))
    target_data, target_num_points = generate_target_data(
        batch_size=10, max_points=100, features=3
    )

    showerdata.save(shower_data, "data/test_integration.h5", overwrite=True)
    showerdata.add_target_dataset("data/test_integration.h5", target_data.shape)
    showerdata.save_target_batch(
        target_data, "data/test_integration.h5", target_num_points
    )

    loaded_showers = showerdata.load("data/test_integration.h5")
    loaded_target, loaded_target_num_points = showerdata.load_target(
        "data/test_integration.h5"
    )

    assert loaded_showers == shower_data, "Loaded shower data should match original"
    assert loaded_target.shape == target_data.shape, (
        "Loaded target data shape should match original"
    )
    assert np.array_equal(loaded_target_num_points, target_num_points), (
        "Loaded target num_points should match original"
    )

    subset_showers = showerdata.load("data/test_integration.h5", start=2, stop=7)
    subset_target, subset_target_num_points = showerdata.load_target(
        "data/test_integration.h5", start=2, stop=7
    )

    assert len(subset_showers) == 5, "Subset should have 5 showers"
    assert subset_target.shape[0] == 5, "Subset target should have 5 entries"
    assert len(subset_target_num_points) == 5, (
        "Subset target num_points should have 5 entries"
    )

    os.remove("data/test_integration.h5")


def test_target_data_edge_cases() -> None:
    os.makedirs("data", exist_ok=True)

    target_data = np.random.uniform(0, 1, size=(5, 20, 3)).astype(np.float32)
    num_points = np.array([0, 5, 0, 10, 15], dtype=np.int32)
    for i in range(5):
        target_data[i, num_points[i] :, :] = 0.0

    showerdata.save_target(
        target_data, "data/test_edge_cases.h5", num_points, overwrite=True
    )
    loaded_data, loaded_num_points = showerdata.load_target("data/test_edge_cases.h5")

    assert np.array_equal(loaded_num_points, num_points), (
        "Zero points should be handled correctly"
    )
    assert loaded_data.shape == target_data.shape, (
        "Shape should be preserved with zero points"
    )

    max_points = 50
    target_data_max = np.random.uniform(0, 1, size=(3, max_points, 4)).astype(
        np.float32
    )
    num_points_max = np.array([max_points, max_points, max_points], dtype=np.int32)

    showerdata.save_target(
        target_data_max, "data/test_edge_cases_max.h5", num_points_max, overwrite=True
    )
    loaded_data_max, loaded_num_points_max = showerdata.load_target(
        "data/test_edge_cases_max.h5"
    )

    assert np.array_equal(loaded_num_points_max, num_points_max), (
        "Maximum points should be handled correctly"
    )
    assert loaded_data_max.shape == target_data_max.shape, (
        "Shape should be preserved with maximum points"
    )

    os.remove("data/test_edge_cases.h5")
    os.remove("data/test_edge_cases_max.h5")
