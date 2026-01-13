# ShowerData
A library to save and load calorimeter shower data in HDF5 format. It stores variable-size point-clouds efficiently and provides easy access to the data.

- [Installation](#installation)
- [Development](#development)

## Installation
You can install the library using pip:

```bash
pip install git+https://github.com/FLC-QU-hep/ShowerData.git
```

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
