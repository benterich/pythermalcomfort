from typing import Union, List

import numpy as np

from pythermalcomfort.utilities import (
    units_converter,
)

def mean_radiant_temperature(
    surface_temps: Union[List[float], np.ndarray],
    angle_factors: Union[List[float], np.ndarray],
    method: str = "linear",
    units: str = "SI",
) -> float:
    """Determines the adaptive thermal comfort based on EN 16798-1 2019 [3]_

    Parameters
    ----------
    tdb : float, int, or array-like
        dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'
    tr : ...

    Returns
    -------
    tmp_cmf : float, int, or array-like
        Comfort temperature at that specific running mean temperature, default in [°C]
        or in [°F]
    acceptability_cat_i : ..

    Notes
    -----
    You can use this function to calculate ..

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import mean_radiant_temperature

    """


    surface_temps = np.array(surface_temps, dtype=float)
    angle_factors = np.array(angle_factors, dtype=float)

    if not np.isclose(np.sum(angle_factors), 1.0):
        raise ValueError("Angle factors must sum to 1.")

    if units.lower() == "ip":
        surface_temps = np.array(
            units_converter(from_units="ip", **{"tmp_surface": surface_temps})
        )

    if method.lower() == "linear":
        mrt = np.sum(angle_factors * surface_temps)
    elif method.lower() == "fourthpower":
        mrt = np.sum(angle_factors * (surface_temps + 273.15) ** 4) ** 0.25 - 273.15
    else:
        raise ValueError("Unsupported method. Choose 'linear' or 'fourthpower'.")

    if units.lower() == "ip":
        mrt = units_converter(from_units="si", **{"tmp_mrt": mrt})[0]

    return mrt