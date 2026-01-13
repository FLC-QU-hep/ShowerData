"""ShowerData - A Python library for shower data storage."""

__author__ = "Thorsten Buss"

from . import observables
from ._version import __version__
from .cluster_module import cluster
from .core import (
    IncidentParticles,
    ShowerDataFile,
    Showers,
    add_target_dataset,
    concatenate,
    create_empty_file,
    get_file_length,
    get_file_shape,
    load,
    load_inc_particles,
    load_target,
    save,
    save_batch,
    save_target,
    save_target_batch,
)
from .filter import filter_showers

__all__ = [
    "observables",
    "__author__",
    "__version__",
    "add_target_dataset",
    "load_target",
    "save_target",
    "save_target_batch",
    "cluster",
    "concatenate",
    "create_empty_file",
    "filter_showers",
    "get_file_length",
    "get_file_shape",
    "IncidentParticles",
    "load",
    "load_inc_particles",
    "save",
    "save_batch",
    "ShowerDataFile",
    "Showers",
]
