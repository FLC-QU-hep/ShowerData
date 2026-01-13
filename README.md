# ShowerData
[![PyPI](https://img.shields.io/pypi/v/showerdata)](https://pypi.org/project/showerdata/)
[![Python Version](https://img.shields.io/pypi/pyversions/showerdata)](https://www.python.org/)
[![License](https://img.shields.io/pypi/l/showerdata)](https://github.com/FLC-QU-hep/ShowerData/blob/main/LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/FLC-QU-hep/ShowerData/publish.yml?label=build&logo=github)](https://github.com/FLC-QU-hep/ShowerData/actions/workflows/publish.yml)
[![Unittests](https://img.shields.io/github/actions/workflow/status/FLC-QU-hep/ShowerData/ci.yml?label=unittests&logo=github)](https://github.com/FLC-QU-hep/ShowerData/actions/workflows/ci.yml)

A library to save and load calorimeter shower data in HDF5 format. It stores variable-size point-clouds efficiently and provides easy access to the data.

- [Installation](#installation)
- [Documentation](#documentation)
- [Development](#development)

## Installation
You can install the library using pip:

```bash
pip install showerdata
```

## Documentation
  The full documentation is available at: [https://flc-qu-hep.github.io/ShowerData/](https://flc-qu-hep.github.io/ShowerData/)

## Development
If you want to contribute to the development of the library, follow these steps to set up your development environment.

### 1. Clone the repository
```bash
git clone https://github.com/FLC-QU-hep/ShowerData.git
cd ShowerData
```

### 2. Install dependencies
Use one of the following methods to install the required dependencies.

#### uv (recommended):
```bash
uv sync --group=dev --group=test --group=doc
source .venv/bin/activate
```

#### pip + venv (alternative):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install --group dev
pip install --group doc
pip install --group test
```

### 3. Setup pre-commit hooks
```bash
pre-commit install
```

### 4. Run unit tests
```bash
pytest
```
