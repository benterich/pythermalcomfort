import numpy as np

from pythermalcomfort.models import mean_radiant_temperature

def test_mean_radiant_temperature():
    test_cases = [
        {
            "surface_temps": [20, 22, 24, 26],
            "angle_factors": [0.25, 0.25, 0.25, 0.25],
            "method": "fourthpower",
            "units": "SI",
        },
        {
            "surface_temps": [20, 22, 24, 26],
            "angle_factors": [0.25, 0.25, 0.25, 0.25],
            "method": "linear",
            "units": "SI",
        },
        {
            "surface_temps": [20, 22, 24, 26],
            "angle_factors": [0.25, 0.25, 0.25, 0.25],
            "method": "fourthpower",
            "units": "IP",
        },
        {
            "surface_temps": [20, 22, 24, 26],
            "angle_factors": [0.25, 0.25, 0.25, 0.25],
            "method": "linear",
            "units": "IP",
        },
    ]

    for test_case in test_cases:
        result = mean_radiant_temperature(
            test_case["surface_temps"],
            test_case["angle_factors"],
            method=test_case["method"],
            units=test_case["units"],
        )

        assert isinstance(result, (float, np.float64)), "Result should be a float or np.float64"
        assert not np.isnan(result), "Result should not be NaN" 

    try:
        mean_radiant_temperature(
            surface_temps=[20, 22, 24],
            angle_factors=[0.4, 0.3, 0.2],
            method="linear",
        )
    except ValueError as e:
        assert str(e) == "Angle factors must sum to 1."

    try:
        mean_radiant_temperature(
            surface_temps=[20, 22, 24],
            angle_factors=[0.33, 0.33, 0.34],
            method="unsupported",
        )
    except ValueError as e:
        assert str(e) == "Unsupported method. Choose 'linear' or 'fourthpower'."