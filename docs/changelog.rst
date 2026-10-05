Changelog
=========

All notable changes to ShowerData will be documented in this file.


[Unreleased]
------------

Added
~~~~~
- ``save``/``load`` aliases for ``save_observables_to_file``/``read_observables_from_file`` in the observables module
- ``-t``/``--threshold`` option for the ``add-observables`` CLI subcommand to set the hit energy threshold
- ``threshold`` argument for ``detector.get_test_geometry``

Changed
~~~~~~~
- CLI returns a non-zero exit code on errors and reports them as warnings on stderr
- Removed upper bound on supported Python version
- Complete type annotations across the codebase, checked with ``ty``

Fixed
~~~~~
- Fixed the behavior of the `shift_layers` function not to apply the box cut when the inverse flag is set
- Incorrect default in the ``--num-layers`` help text of the ``add-observables`` subcommand
- Incorrect ``save_target_batch`` docstring example


[0.6.1] - 2026-01-30
--------------------
Fixed
~~~~~
- Make ShowerDataFile iterable over showers as expected
- Fix urls in pyproject.toml
- Improve error messages when initializing Showers object with invalid data


[0.6.0] - 2026-01-13
--------------------

Added
~~~~~
- Filter subcommand to filter hits in showers based on energy threshold and spatial region
- Save some basic metadata (e.g., clustering status, ShowerData version) in HDF5 files
- get_file_length function that also works for files only containing incident particles

Changed
~~~~~~~
- Observable calculation functions takes detector thresholds into account

Fixed
~~~~~
- Incorrect and inconsistent version number in some places


[0.5.0] - 2025-11-18
--------------------

Added
~~~~~
- Support python 3.14
- New observable calculations: energy per radial bin, center of energy
- '--overwrite' flag for 'add-observables' CLI subcommand

Changed
~~~~~~~
- More consistent naming of observables
- Stricter type hints across the codebase
- Improved error messages

Fixed
~~~~~
- Bug in documentation code example


[0.4.0] - 2025-09-30
--------------------

Added
~~~~~
- Clustering algorithms for shower data analysis
- Command-Line subcommand for clustering showers
- Support for storing target data in HDF5 files


[0.3.0] - 2025-09-16
--------------------

Added
~~~~~
- Core functionality for HDF5 shower data storage
- Basic observable calculation functions
- Command-line subcommand for adding observables
- Command-line subcommand for shifting showers


[0.2.0] - 2025-08-25
--------------------

Added
~~~~~
- Basic shower data handling capabilities
- Support for variable-size point clouds
- HDF5 file format support
- Initial API design
- Command-Line subcommand for shuffling data
