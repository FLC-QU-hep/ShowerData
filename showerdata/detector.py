"""\
Module defining a simple data class for some detector geometry
information and functions initializing common detector geometries.
"""

from dataclasses import dataclass

__all__ = ["DetectorGeometry", "get_ILD_geometry"]


@dataclass(frozen=True, eq=True, repr=False)
class DetectorGeometry:
    calo_surface: float
    num_layers: int
    num_layers_ecal: int
    num_layers_hcal: int
    layer_thickness_ecal: float
    layer_thickness_hcal: float
    ecal_cell_size: float
    hcal_cell_size: float
    layer_bottom_pos: tuple[float, ...]
    ecal_threshold: float = 0.0
    hcal_threshold: float = 0.0
    name: str = "Detector"

    def __repr__(self) -> str:
        return f"Geometry({self.name})"


def get_ILD_geometry() -> DetectorGeometry:
    return DetectorGeometry(
        calo_surface=1804.0,
        num_layers=78,
        num_layers_ecal=30,
        num_layers_hcal=48,
        layer_thickness_ecal=0.525,
        layer_thickness_hcal=1.2,
        ecal_cell_size=5.0,
        hcal_cell_size=30.0,
        ecal_threshold=0.1,
        hcal_threshold=0.25,
        layer_bottom_pos=(
            1811.34,
            1814.46,
            1823.81,
            1826.94,
            1836.28,
            1839.41,
            1848.75,
            1851.88,
            1861.22,
            1864.34,
            1873.69,
            1876.81,
            1886.16,
            1889.29,
            1898.63,
            1901.76,
            1911.1,
            1914.22,
            1923.57,
            1926.69,
            1938.14,
            1943.36,
            1954.81,
            1960.04,
            1971.48,
            1976.7,
            1988.15,
            1993.38,
            2004.82,
            2010.05,
            2081.0,
            2107.5,
            2134.0,
            2160.5,
            2187.0,
            2213.5,
            2240.0,
            2266.5,
            2293.0,
            2319.5,
            2346.0,
            2372.5,
            2399.0,
            2425.5,
            2452.0,
            2478.5,
            2505.0,
            2531.5,
            2558.0,
            2584.5,
            2611.0,
            2637.5,
            2664.0,
            2690.5,
            2717.0,
            2743.5,
            2770.0,
            2796.5,
            2823.0,
            2849.5,
            2876.0,
            2902.5,
            2929.0,
            2955.5,
            2982.0,
            3008.5,
            3035.0,
            3061.5,
            3088.0,
            3114.5,
            3141.0,
            3167.5,
            3194.0,
            3220.5,
            3247.0,
            3273.5,
            3300.0,
            3326.5,
        ),
        name="ILD",
    )


def get_ILD_unclustered_geometry() -> DetectorGeometry:
    geometry = get_ILD_geometry()
    geometry = DetectorGeometry(
        calo_surface=geometry.calo_surface,
        num_layers=geometry.num_layers,
        num_layers_ecal=geometry.num_layers_ecal,
        num_layers_hcal=geometry.num_layers_hcal,
        layer_thickness_ecal=geometry.layer_thickness_ecal,
        layer_thickness_hcal=geometry.layer_thickness_hcal,
        ecal_cell_size=0.0,
        hcal_cell_size=0.0,
        layer_bottom_pos=geometry.layer_bottom_pos,
        ecal_threshold=0.0,
        hcal_threshold=0.0,
        name="ILD_unclustered",
    )
    return geometry


def get_test_geometry(num_layers: int) -> DetectorGeometry:
    return DetectorGeometry(
        calo_surface=100.0,
        num_layers=num_layers,
        num_layers_ecal=num_layers,
        num_layers_hcal=0,
        layer_thickness_ecal=1.0,
        layer_thickness_hcal=1.0,
        ecal_cell_size=10.0,
        hcal_cell_size=10.0,
        layer_bottom_pos=tuple(100.0 + i * 1.0 for i in range(num_layers)),
        name="TestDetector",
    )
