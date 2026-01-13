from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("showerdata")
except PackageNotFoundError:
    __version__ = ""  # package is not installed
