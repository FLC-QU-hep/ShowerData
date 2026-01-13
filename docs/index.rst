ShowerData Documentation
========================

ShowerData is a Python library for efficient storage and retrieval of calorimeter shower data in HDF5 format.
It is designed for the training of fast machine learning-based surrogate models of particle showers simulations.
One of its key advantages is the consistent use of type hints throughout the codebase.

Features
--------

* **Efficient HDF5 Storage**: Storing variable-length point clouds without padding
* **Easy Data Access**: Simple API for loading and saving shower data
* **Observable Calculations**: Functions to compute basic observables
* **Command-Line Interface**: Tools for shuffling, adding observables, shifting, and clustering showers

Quick Start
-----------

Install ShowerData using pip:

.. code-block:: bash

   pip install git+https://github.com/FLC-QU-hep/ShowerData.git

Load and work with shower data:

.. code-block:: python

   import showerdata
   import numpy as np

   # Load first 1000 showers from an HDF5 file
   showers = showerdata.load("path/to/your/file.h5", stop=1000)

   # Access numpy array of shower
   hits = showers.points
   inc_pdg = showers.pdg
   inc_energies = showers.energies
   inc_directions = showers.directions

   # Calculate observables for first 10 showers
   num_points = showerdata.observables.calc_num_points_per_layer(showers[:10])
   layer_energies = showerdata.observables.calc_energy_per_layer(showers[:10])

   # Load precomputed observables if available
   # Note: running "showerdata add-observables" first is recommended
   observables = showerdata.observables.read_observables_from_file(
      path="path/to/your/file.h5",
      stop=10,
      observables=["num_points_per_layer", "energy_per_layer"]
   )

   assert np.all(num_points == observables["num_points_per_layer"])
   assert np.all(layer_energies == observables["energy_per_layer"])

   showers.save("new_file1.h5")  # Save showers to a new file
   showerdata.save(showers, "new_file2.h5") # Equivalent way to save

   # Create a Showers object from numpy arrays
   showers = showerdata.Showers(
      points=hits,
      pdg=inc_pdg,
      energies=inc_energies,
      directions=inc_directions
   )

Use the command-line interface:

.. code-block:: bash

   # Shuffle multiple HDF5 files into one
   showerdata shuffle file1.h5 file2.h5 shuffled_file.h5 --overwrite

   # Add observables to an existing HDF5 file
   showerdata add-observables file.h5 --batch-size 1000

   # Shift showers in an existing HDF5 file to counteract the incident angle
   showerdata shift input_file.h5 output_file.h5

   # Shift back to original positions
   showerdata shift output_file.h5 restored_file.h5 --inverse

   # Cluster showers
   showerdata cluster input_file.h5 output_file.h5 --processes 8

   # Filter hits in showers based on energy threshold and spatial region
   showerdata filter input_file.h5 output_file.h5 -r 200 -d

   # List available CLI commands
   showerdata

   # Get help on a specific command
   showerdata <subcommand> --help

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 1
   :caption: API Reference:

   core
   observables

.. toctree::
   :maxdepth: 1
   :caption: Development:

   changelog

Indices
-------
* :ref:`genindex`
